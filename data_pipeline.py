import random
from typing import Dict, Optional
import numpy as np
from data_parser import PHHSParser, ParseStats, ActionParser, PrintStats


def Playerfilter(hands, player: str, position: str):           #filters based on specific player

    positionId = 0 if position == 'p1' else 1
    out = []
    for hand in hands:
        if player in hand['players']:

            if hand['players'].index(player) == positionId:
                out.append(hand)
    return out


def BuildSequence(hands) -> Dict[int, list]:                      #builds sequences based on the parsed data from the parser

    sequences = {}
    for idx, hand in enumerate(hands):
        seq = [ActionParser(a) for a in hand['actions']]
   
        sequences[idx] = [a for a in seq if a is not None]
    return sequences


def Printsample(seq, n: int = 10):                                 #prints the first few actions of a sequence

    print(f"\nSample sequence (hand 0, first {n} actions):")
    for i, action in enumerate(seq[:n]):
        t = action.get('type')
        if t == 'deal':
            print(f"  {i}: Deal {action['player']}: {action['cards']}")
        elif t == 'board':
            print(f"  {i}: Board: {action['cards']}")
        elif t == 'action':
            print(f"  {i}: {action['player']} {action['action']} {action.get('amount', '')}")


def Completebuild(folder: str,
                          player: Optional[str] = None,
                          position: str = 'p1',
                          clusternumber: int = 3,
                          maxhands: Optional[int] = None,
                          maxiterations: int = 20,
                          output: str = './poker_analysis',
                          random: int = 42):


    random.seed(random)
    np.random.seed(random)

    print("=" * 80)
    print("POKER PLAYING STYLE CLUSTERING - COMPLETE PIPELINE")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Folder: {folder}")
    print(f"  Player: {player if player else 'ALL'}")
    print(f"  Position: {position}")
    print(f"  Clusters: {clusternumber}")
    print(f"  Max hands: {maxhands if maxhands else 'ALL'}")
    print(f"  Output: {output}")


    print("\n" + "=" * 80)
    print("STEP 1: PARSING PHHS FILES")
    print("=" * 80)

    all_hands = PHHSParser.Folderparse(folder, pattern="*.phhs")              #parses all phhs files
    if not all_hands:
        print("ERROR: No hands found!")
        return None

    PrintStats(ParseStats(all_hands))


    print("\n" + "=" * 80)
    print("STEP 2: FILTERING HANDS")
    print("=" * 80)

    if player:
        filteredhands = Playerfilter(all_hands, player, position)                        #applies filter if necessary
        print(f"Filtered to {len(filteredhands)} hands where {player} is {position}")
    else:
        filteredhands = all_hands
        print(f"Using all {len(filteredhands)} hands (no player filter)")

    if maxhands and len(filteredhands) > maxhands:
        filteredhands = filteredhands[:maxhands]
        print(f"Limited to first {maxhands} hands")

    if not filteredhands:
        print("ERROR: No hands remaining after filtering!")
        return None


    print("\n" + "=" * 80)
    print("STEP 3: EXTRACTING ACTION SEQUENCES")                                    #converts raw data to action sequences
    print("=" * 80)

    max_actions = max(len(h['actions']) for h in filteredhands)
    print(f"Max actions in any hand: {max_actions}")

    handsequences = BuildSequence(filteredhands)
    print(f"Extracted {len(handsequences)} hand sequences")


    if handsequences:
        Printsample(handsequences[0])


    p1_action_count = sum(
        1
        for seq in handsequences.values()
        for a in seq
        if a.get('type') == 'action' and a.get('player') == 'p1'
    )
    print(f"\nTotal P1 actions across all hands: {p1_action_count}")

    return {
        'handsequences': handsequences,
        'filteredhands': filteredhands,
        'config': {
            'clusternumber':     clusternumber,
            'maxiterations': maxiterations,
            'output':     output,
        },
    }
