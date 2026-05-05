"""
Plotting utilities for poker clustering visualisation.

Produces a 1×3 figure each iteration showing how cluster centroids move
over time across three complementary stat pairs:

  Plot 1: VPIP vs PFR
           (Voluntarily Put money In Pot vs Pre-Flop Raise rate — the two
            most common top-level style indicators in poker analytics.)

  Plot 2: Mean E[HS²] when folding vs Mean E[HS²] when raising
           (Captures how strong a player's hand typically is at the moment
            they fold or raise.  Tight players fold weak hands and raise
            strong ones; loose players do both with medium-strength hands.)

  Plot 3: Mean E[HS²] when folding vs Fold-to-aggression rate
           (Shows whether a cluster folds to pressure with genuinely weak
            hands or capitulates too easily.)

Every past iteration's data is retained in a class-level history so plots
show the full progression: old points fade with age and arrows connect
successive positions of each cluster.
"""

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# treys provides a fast 7-card hand evaluator (returns a rank where
# lower = better).
from treys import Card as TreysCard, Evaluator as TreysEvaluator


# ===========================================================================
# E[HS²] CALCULATION (cached)
# ===========================================================================

class EHSCalculator:
    """
    Compute E[HS²] for a hand at a given board state via Monte Carlo
    rollout to the river against a uniformly random opponent.

    E[HS²] — "expected hand strength squared" — is the expected value of
    (hand strength)² over all possible remaining community cards and
    opponent hole cards.  Hand strength for a single runout is
    P(beat the random opponent at showdown).

    Squaring before averaging preserves variance information that plain
    E[HS] discards.  This means high-potential drawing hands score closer
    to made hands than to medium-strength ones, which is a more useful
    grouping for strategy analysis (Johanson 2007).

    Results are cached by (hole cards, board, num_samples) so repeated
    queries for the same game state are free.
    """

    _evaluator = TreysEvaluator()
    _cache: Dict[Tuple, float] = {}

    # Number of random rollouts per evaluation.  200 gives a good
    # accuracy-vs-speed tradeoff for clustering purposes.
    NUM_SAMPLES = 200

    @staticmethod
    def _parse_cards(cards_str: str) -> List[int]:
        """Convert a concatenated card string to treys integer card list."""
        if not cards_str:
            return []
        return [
            TreysCard.new(cards_str[i:i + 2])
            for i in range(0, len(cards_str) - 1, 2)
        ]

    @classmethod
    def compute_ehs_squared(cls, hole_cards_str: str, board_cards_str: str = '',
                            num_samples: Optional[int] = None) -> float:
        """
        Compute E[HS²] for a hand at the current board state.

        Parameters
        ----------
        hole_cards_str : str
            Two-card string, e.g. 'AsKh'.
        board_cards_str : str
            Community cards seen so far (0–5 cards), e.g. 'AcKdQh'.
        num_samples : int or None
            Override the default number of Monte Carlo rollouts.

        Returns a float in [0, 1].  0.5 is used as a neutral fallback for
        malformed input.
        """
        if num_samples is None:
            num_samples = cls.NUM_SAMPLES

        try:
            hole = cls._parse_cards(hole_cards_str)
            board = cls._parse_cards(board_cards_str)
        except Exception:
            return 0.5  # malformed cards → neutral

        if len(hole) != 2:
            return 0.5

        # Cache lookup: frozensets make order-insensitive keys.
        cache_key = (frozenset(hole), frozenset(board), num_samples)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # Build remaining deck (52 cards minus what's already in play).
        all_cards = [TreysCard.new(r + s)
                     for r in '23456789TJQKA' for s in 'shdc']
        used = set(hole) | set(board)
        remaining_deck = [c for c in all_cards if c not in used]

        # Each sample needs enough cards to complete the board (0–5) plus
        # two opponent hole cards.
        community_needed = 5 - len(board)
        cards_per_sample = community_needed + 2
        if len(remaining_deck) < cards_per_sample:
            cls._cache[cache_key] = 0.5
            return 0.5

        ehs_squared_sum = 0.0
        # Deterministic RNG seeded from the hand itself so results are
        # reproducible across runs.
        rng = random.Random(hash(cache_key) & 0xFFFFFFFF)

        for _ in range(num_samples):
            sample = rng.sample(remaining_deck, cards_per_sample)
            future_board = list(board) + sample[:community_needed]
            opp_hole = sample[community_needed:community_needed + 2]

            # treys evaluate(): lower rank = stronger hand.
            our_rank = cls._evaluator.evaluate(hole, future_board)
            opp_rank = cls._evaluator.evaluate(opp_hole, future_board)

            if our_rank < opp_rank:
                hs = 1.0    # we win
            elif our_rank > opp_rank:
                hs = 0.0    # we lose
            else:
                hs = 0.5    # tie

            ehs_squared_sum += hs * hs

        result = ehs_squared_sum / num_samples
        cls._cache[cache_key] = result
        return result


# ===========================================================================
# PER-CLUSTER STATS
# ===========================================================================

class PokerStatsCalculator:
    """
    Compute aggregate poker statistics for each cluster.

    These stats are used by the plotter to position clusters in 2-D space
    and are also useful as a human-readable summary of each cluster's
    playing style.
    """

    @staticmethod
    def calculate_stats_per_player(sequences: Dict[int, List[Dict]],
                                   clusters: Dict[int, int],
                                   target_player: str = 'p1') -> Dict[int, Dict]:
        """
        For each cluster compute:

          VPIP  — Voluntarily Put money In Pot: fraction of hands where the
                  target player voluntarily invested chips pre-flop (call or
                  raise, excluding the forced blind).
          PFR   — Pre-Flop Raise: fraction of hands where the target player
                  raised before the flop.
          Aggression factor — post-flop (bets + raises) / calls.
          Mean E[HS²] when folding — average hand strength at fold moments.
          Mean E[HS²] when raising — average hand strength at raise moments.
          Fold-to-aggression — how often the player folds when facing a
                               bet or raise from the opponent.

        Parameters
        ----------
        sequences : dict
            hand_index → action sequence.
        clusters : dict
            hand_index → cluster_id.
        target_player : str
            Which player's perspective to analyse ('p1' or 'p2').

        Returns
        -------
        dict mapping cluster_id → stats dict.
        """
        # Group sequences by cluster.
        cluster_seqs: Dict[int, List[List[Dict]]] = defaultdict(list)
        for seq_id, cid in clusters.items():
            if seq_id in sequences:
                cluster_seqs[cid].append(sequences[seq_id])

        stats: Dict[int, Dict] = {}

        for cid, seqs in cluster_seqs.items():
            total_hands = len(seqs)
            vpip_count = pfr_count = 0
            total_bets_raises = total_calls = 0

            ehs2_on_fold: List[float] = []
            ehs2_on_raise: List[float] = []

            faced_aggression = folded_to_aggression = 0

            for seq in seqs:
                voluntary_put_money = preflop_raise = flop_seen = False
                first_action_by_player = True
                hole_cards = ''
                board_so_far = ''
                opponent_just_aggressed = False

                for action in seq:
                    a_type = action.get('type')

                    # Record target player's hole cards when dealt.
                    if a_type == 'deal' and action.get('player') == target_player:
                        hole_cards = action.get('cards', '')
                        continue

                    # Track community board cards as they appear.
                    if a_type == 'board':
                        flop_seen = True
                        board_so_far += action.get('cards', '')
                        continue

                    if a_type != 'action':
                        continue

                    actor = action.get('player')
                    a = action.get('action')
                    amount = action.get('amount', 0)

                    # --- Opponent actions ---
                    # We only need to know if the opponent just bet/raised
                    # (for the fold-to-aggression metric).
                    if actor != target_player:
                        if a in ('raise', 'bet'):
                            opponent_just_aggressed = True
                        elif a in ('call', 'check', 'fold'):
                            opponent_just_aggressed = False
                        continue

                    # --- Target player actions ---

                    # Pre-flop: determine VPIP and PFR.
                    if not flop_seen:
                        if first_action_by_player:
                            if a in ('raise', 'bet'):
                                voluntary_put_money = True
                                preflop_raise = True
                            elif a == 'call' and amount > 0:
                                voluntary_put_money = True
                            first_action_by_player = False
                        else:
                            if a in ('raise', 'bet', 'call'):
                                voluntary_put_money = True
                            if a in ('raise', 'bet'):
                                preflop_raise = True

                    # Post-flop: accumulate aggression factor numerator /
                    # denominator.
                    if flop_seen:
                        if a in ('raise', 'bet'):
                            total_bets_raises += 1
                        elif a == 'call':
                            total_calls += 1

                    # Record E[HS²] at the moment the player folds or raises
                    # (only if we know their hole cards).
                    if a == 'fold' and hole_cards:
                        ehs2_on_fold.append(
                            EHSCalculator.compute_ehs_squared(hole_cards,
                                                              board_so_far))
                    elif a == 'raise' and hole_cards:
                        ehs2_on_raise.append(
                            EHSCalculator.compute_ehs_squared(hole_cards,
                                                              board_so_far))

                    # Fold-to-aggression: did the player fold when the
                    # opponent had just bet or raised?
                    if opponent_just_aggressed:
                        faced_aggression += 1
                        if a == 'fold':
                            folded_to_aggression += 1
                        opponent_just_aggressed = False

                if voluntary_put_money:
                    vpip_count += 1
                if preflop_raise:
                    pfr_count += 1

            # Compute final rates for this cluster.
            vpip = vpip_count / total_hands if total_hands else 0.0
            pfr = pfr_count / total_hands if total_hands else 0.0

            if total_calls == 0 and total_bets_raises > 0:
                aggression = 100.0   # infinite aggression, capped
            elif total_calls > 0:
                aggression = total_bets_raises / total_calls
            else:
                aggression = 0.0

            stats[cid] = {
                'vpip':       vpip,
                'pfr':        pfr,
                'aggression': aggression,
                'num_hands':  total_hands,
                'mean_ehs2_fold':      (float(np.mean(ehs2_on_fold))
                                        if ehs2_on_fold else 0.0),
                'mean_ehs2_raise':     (float(np.mean(ehs2_on_raise))
                                        if ehs2_on_raise else 0.0),
                'fold_to_aggression':  (folded_to_aggression / faced_aggression
                                        if faced_aggression else 0.0),
                # Debug counters (useful for sanity-checking).
                'debug_vpip_count':        vpip_count,
                'debug_pfr_count':         pfr_count,
                'debug_bets_raises':       total_bets_raises,
                'debug_calls':             total_calls,
                'debug_fold_count':        len(ehs2_on_fold),
                'debug_raise_count':       len(ehs2_on_raise),
                'debug_faced_aggression':  faced_aggression,
            }

        return stats


# ===========================================================================
# PROGRESSIVE PLOTTER
# ===========================================================================

class ClusterPlotter:
    """
    Plot cluster evolution across iterations.

    Maintains a class-level history so that each call to plot_clusters()
    draws *every* iteration's data points.  Older points fade with age
    and arrows track each cluster's movement between successive iterations.

    Call ClusterPlotter.reset() to clear history before a new run (this
    also happens automatically when iteration == 0).
    """

    # Colour palette — one per cluster, wraps around for k > 10.
    COLOURS = [
        '#1f77b4', '#9467bd', '#ff7f0e', '#d62728', '#2ca02c',
        '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
    ]

    # History: list of (iteration_number, cluster_stats_dict).
    _history: List[Tuple[int, Dict[int, Dict]]] = []

    @classmethod
    def reset(cls):
        """Clear all accumulated history (for a fresh run)."""
        cls._history = []

    @classmethod
    def plot_clusters(cls,
                      sequences: Dict[int, List[Dict]],
                      clusters: Dict[int, int],
                      iteration: int,
                      output_path: Optional[str] = None,
                      target_player: str = 'p1'):
        """
        Compute stats for the current iteration, append to history,
        and produce a 1×3 figure showing the full progression.

        Parameters
        ----------
        sequences : dict
            All hand sequences (hand_index → action list).
        clusters : dict
            Current cluster assignments (hand_index → cluster_id).
        iteration : int
            Current iteration number (0 = initial state).
        output_path : str or None
            If given, save the figure as a PNG to this path.
        target_player : str
            Which player's stats to compute ('p1' or 'p2').
        """
        # Compute per-cluster poker stats for this iteration.
        cluster_stats = PokerStatsCalculator.calculate_stats_per_player(
            sequences, clusters, target_player
        )

        # Iteration 0 restarts history.  Otherwise prune any stale future
        # entries (from a partial prior run) and append the new snapshot.
        if iteration == 0:
            cls._history = []
        cls._history = [(it, st) for (it, st) in cls._history if it < iteration]
        cls._history.append((iteration, cluster_stats))

        cls._print_debug(iteration, cluster_stats)

        # Determine every cluster id that has ever appeared.
        unique_clusters = sorted({cid for _, snap in cls._history
                                  for cid in snap})

        # Create the 1×3 subplot figure.
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Cluster evolution — through iteration {iteration}',
                     fontsize=15, fontweight='bold')

        cls._plot_progression(
            axes[0], unique_clusters,
            x_key='vpip', y_key='pfr',
            x_label='VPIP', y_label='PFR',
            title='VPIP vs PFR',
        )
        cls._plot_progression(
            axes[1], unique_clusters,
            x_key='mean_ehs2_fold', y_key='mean_ehs2_raise',
            x_label='Mean E[HS²] when folding',
            y_label='Mean E[HS²] when raising',
            title='Hand strength: folds vs raises',
        )
        cls._plot_progression(
            axes[2], unique_clusters,
            x_key='mean_ehs2_fold', y_key='fold_to_aggression',
            x_label='Mean E[HS²] when folding',
            y_label='Fold-to-aggression rate',
            title='Fold strength vs fold-to-aggression',
        )

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            print(f'Saved plot to {output_path}')

        plt.show()
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @classmethod
    def _plot_progression(cls, ax, unique_clusters,
                          x_key, y_key, x_label, y_label, title):
        """
        Draw one subplot showing every cluster's trajectory through
        (x_key, y_key) space across all recorded iterations.

        Older points are drawn smaller and more transparent.  Arrows
        connect successive iterations for each cluster.  The latest
        iteration's points are large and opaque.
        """
        history = cls._history
        if not history:
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            return

        n_iters = len(history)
        latest_iter = history[-1][0]

        for cid in unique_clusters:
            colour = cls.COLOURS[cid % len(cls.COLOURS)]

            # Collect this cluster's coordinates across iterations.
            xs, ys, iters = [], [], []
            for it, snap in history:
                if cid in snap:
                    xs.append(snap[cid][x_key])
                    ys.append(snap[cid][y_key])
                    iters.append(it)

            if not xs:
                continue

            # Draw the path line and directional arrows.
            if len(xs) > 1:
                ax.plot(xs, ys, color=colour, alpha=0.35,
                        linewidth=1.2, zorder=2)
                for k in range(len(xs) - 1):
                    ax.annotate(
                        '',
                        xy=(xs[k + 1], ys[k + 1]),
                        xytext=(xs[k], ys[k]),
                        arrowprops=dict(
                            arrowstyle='->', color=colour,
                            alpha=0.55, lw=1.2,
                            shrinkA=4, shrinkB=4,
                        ),
                        zorder=3,
                    )

            # Draw per-iteration scatter points.
            for k, (x, y, it) in enumerate(zip(xs, ys, iters)):
                if it == latest_iter:
                    # Current iteration: large, bold, labelled.
                    ax.scatter(x, y, color=colour, s=140,
                               edgecolors='white', linewidths=1.8,
                               alpha=0.95, zorder=5,
                               label=f'Cluster {cid}')
                elif k == 0:
                    # Starting point: hollow ring to mark the origin.
                    ax.scatter(x, y, facecolors='none', edgecolors=colour,
                               s=90, linewidths=1.6, alpha=0.6, zorder=4)
                else:
                    # Intermediate point: small, fades in over time.
                    age_frac = (k + 1) / max(n_iters, 1)
                    ax.scatter(x, y, color=colour, s=45,
                               alpha=0.15 + 0.5 * age_frac,
                               edgecolors='none', zorder=4)

        # Axis cosmetics.
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best', fontsize=9, framealpha=0.85)

        # Auto-scale with a 20 % pad so points aren't clipped at edges.
        all_x = [snap[cid][x_key] for _, snap in history for cid in snap]
        all_y = [snap[cid][y_key] for _, snap in history for cid in snap]
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_pad = max((x_max - x_min) * 0.20, 0.02)
            y_pad = max((y_max - y_min) * 0.20, 0.02)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

    @staticmethod
    def _print_debug(iteration: int, stats: Dict[int, Dict]):
        """Log cluster stats to the console for debugging."""
        print('\n' + '=' * 60)
        print(f'CLUSTER STATS — iteration {iteration}')
        print('=' * 60)
        for cid in sorted(stats):
            s = stats[cid]
            print(
                f"Cluster {cid} (n={s['num_hands']}): "
                f"VPIP={s['vpip']:.3f}, PFR={s['pfr']:.3f}, "
                f"Aggr={s['aggression']:.2f}"
            )
            print(
                f"   E[HS²] fold={s['mean_ehs2_fold']:.3f} "
                f"(n={s['debug_fold_count']}),  "
                f"E[HS²] raise={s['mean_ehs2_raise']:.3f} "
                f"(n={s['debug_raise_count']}),  "
                f"fold-to-aggression={s['fold_to_aggression']:.3f} "
                f"(n={s['debug_faced_aggression']})"
            )
        print('=' * 60)
