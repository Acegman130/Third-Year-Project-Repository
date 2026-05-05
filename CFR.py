import random
import numpy as np
from collections import Counter
from itertools import combinations
import pickle
import os
import time
import gc
 
 
def Hand_catgory(HoleCard1, HoleCard2):          #Pretty much slims the preflop hands down from like 1326 to 169 cause suits dont matter

    rank1, rank2 = HoleCard1 % 100, HoleCard2 % 100
    suit1, suit2 = HoleCard1 // 100, HoleCard2 // 100

    highrank, lowwrank = max(rank1, rank2), min(rank1, rank2)
 
    if rank1 == rank2:
        return ('pair', highrank, lowwrank, False)
   
    elif suit1 == suit2:
        return ('suited', highrank, lowwrank, True)
    else:
        return ('offsuit', highrank, lowwrank, False)
 
 
 
def HandRanker(holecards, commcards):                   #Ranks hands by giving them a numeric score. Follows the regular hand rankings so the score for triples cant be higher than pair

    combinedcards = holecards + commcards
    suits = [c // 100 for c in combinedcards]          #gets suits and ranks from combinned hole annd community cards

    ranks = [c % 100  for c in combinedcards]

 
    Flushsuit = None
    for suit, count in Counter(suits).items():          #checks if there is a flush
        if count >= 5:
            Flushsuit = suit
            break
 


    Flushranks = []
    if Flushsuit is not None:
        Flushranks = sorted([c % 100 for c in combinedcards if c // 100 == Flushsuit], reverse=True)                #gets cards in flush to determine how high it is for scoring
 
                                                                                                                       #e.g. Ace flush better thann seven flush
    indivdualranks = sorted(set(ranks), reverse=True)         #che3ck for straight
    highest_inn_straight = 0
    for i in range(len(indivdualranks) - 4):
        if indivdualranks[i] - indivdualranks[i + 4] == 4:   #sorts ranks then if dif greater than 4 then cant be straight

            highest_inn_straight = indivdualranks[i]
            break

    if 14 in indivdualranks and {2, 3, 4, 5}.issubset(set(indivdualranks)):         #unique enndge case where Ace acts as one so check for that
        if highest_inn_straight == 0:
            highest_inn_straight = 5  
 
    if Flushsuit is not None and len(Flushranks) >= 5:      #specific section for straight fllush
        sf_high = 0
        for i in range(len(Flushranks) - 4):
            if Flushranks[i] - Flushranks[i + 4] == 4:
                sf_high = Flushranks[i]
                break

        if 14 in Flushranks and {2, 3, 4, 5}.issubset(set(Flushranks)):
            if sf_high == 0:
                sf_high = 5

        if sf_high > 0:
            return 9000 + sf_high
 
    CardNumbers = Counter(ranks)                                                               #sectionn for hannds that have count based stuff e.g trips and pair
    card_counts = sorted(CardNumbers.items(), key=lambda x: (x[1], x[0]), reverse=True)
 

    if card_counts[0][1] == 4:                                          #four of kind
        return 8000 + card_counts[0][0] * 15 + card_counts[1][0]
 


    if card_counts[0][1] == 3 and card_counts[1][1] >= 2:                   #full house
        return 7000 + card_counts[0][0] * 15 + card_counts[1][0]
 

    if Flushsuit is not None:                                           #flush score calc
        score = 6000
        for i, rank in enumerate(Flushranks[:5]):
            score += rank * (15 ** (4 - i))

        return min(score, 6999)                           #cap so no overflow into full house section
 

    if highest_inn_straight > 0:                        #straight
        return 5000 + highest_inn_straight
 

    if card_counts[0][1] == 3:                                                      #trips
        kickers = sorted([r for r, c in card_counts[1:]], reverse=True)[:2]
        score = 4000 + card_counts[0][0] * 15
        if len(kickers) >= 2:
            score += kickers[0] * 0.1 + kickers[1] * 0.01                   #extra board cards (high card deciders)
        return score
 



    if card_counts[0][1] == 2 and card_counts[1][1] == 2:                                      #two pair
        return 3000 + card_counts[0][0] * 15 + card_counts[1][0] + card_counts[2][0] * 0.01
 

    if card_counts[0][1] == 2:
        kickers = sorted([r for r, c in card_counts[1:]], reverse=True)[:3]                 #pair
        score = 2000 + card_counts[0][0] * 15
        for i, k in enumerate(kickers):
            score += k * (0.1 ** (i + 1))
        return score
 


    HighCard = sorted(ranks, reverse=True)[:5]                  #high card
    score = 1000 + HighCard[0] * 15
    for i, c in enumerate(HighCard[1:], 1):
        score += c * (0.1 ** i)
    return score
 
 
 
def PreflopStrenngthCalc(card1, card2, num_simulations=1000):                 #calculates the hand strength for preflop by takinng random samples
                                                                            #need individual cal for preflop cause very hard to simulate all board runs so need sampling for relative streength
    deck = [s * 100 + r for s in range(4) for r in range(2, 15)]         #build deck of 52 cards
    used = {card1, card2}
    unused = [c for c in deck if c not in used]
 
    wins = draws = losses = 0
    for _ in range(num_simulations):
        sample = random.sample(unused, 7)
        Ourhand = HandRanker([card1, card2], sample[2:7])
        OpponentHand = HandRanker(sample[:2], sample[2:7])

        if Ourhand > OpponentHand: wins += 1
        elif Ourhand == OpponentHand: draws += 1
        else: losses += 1
    return (wins + draws * 0.5) / num_simulations
 
 
 
def EHSCalc(ourcards, communitycards, samples=100):                                 #EHS calculator to be used for bucketing  
                                                                                        #EHS= HS * (1-NPOT) + (1-HS) * PPOT
                                                                                           
    deck = [s * 100 + r for s in range(4) for r in range(2, 15)]
    used = set(ourcards + communitycards)
    unused = [c for c in deck if c not in used]
 


    if len(communitycards) == 5:                                     #River section for just HS cause no potential left
        sample_size = min(50, len(unused) // 2)
        ahead = tied = behind = 0
        Ourrank = HandRanker(ourcards, communitycards)
        for _ in range(sample_size):
            Opponent = random.sample(unused, 2)
            opprank = HandRanker(Opponent, communitycards)

            if Ourrank > opprank: ahead += 1
            elif Ourrank == opprank: tied += 1
            else: behind += 1
        total = ahead + tied + behind
        return (ahead + tied / 2) / total if total else 0.5
 

    AHEAD, TIED, BEHIND = 0, 1, 2                              #Preriver section
    HP = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    HPTotal = [0, 0, 0]
    ahead = tied = behind = 0

    Ourrank = HandRanker(ourcards, communitycards)
    nextcards = 5 - len(communitycards)
 

    num_opp = samples                             #sample conntrol
    num_board = 15 if len(communitycards) == 3 else 20
 


    for _ in range(num_opp):
        if len(unused) < 2:
            continue
        Opponent = random.sample(unused, 2)
        opprank = HandRanker(Opponent, communitycards)
 

        if Ourrank > opprank: index = AHEAD;  ahead += 1                   #current Hand strength
        elif Ourrank == opprank: index = TIED;   tied += 1
        else: index = BEHIND; behind += 1
        HPTotal[index] += 1
 

        if nextcards > 0:
            remaining = [c for c in unused if c not in Opponent]                  #see how future strength can improve or decreas
            if len(remaining) >= nextcards:
                for _ in range(num_board):
                    future = random.sample(remaining, nextcards)
                    final_board = communitycards + future

                    CFRbest = HandRanker(ourcards, final_board)
                    Opponentbest = HandRanker(Opponent,      final_board)
                    if   CFRbest >  Opponentbest: HP[index][AHEAD] += 1

                    elif CFRbest == Opponentbest: HP[index][TIED] += 1
                    else:                      HP[index][BEHIND] += 1
 


    total = ahead + tied + behind
    if total == 0:
        return 0.5
    HS = (ahead + tied / 2) / total

    PPOTnumerator = HP[BEHIND][AHEAD] + HP[BEHIND][TIED] / 2 + HP[TIED][AHEAD] / 2            #positive potential PPOT calcs
    PPotDenominator = sum(HP[BEHIND]) + sum(HP[TIED])
    Ppot = max(0.0, min(1.0, PPOTnumerator / PPotDenominator if PPotDenominator > 0 else 0.0))
 

    NPOTNumerator = HP[AHEAD][BEHIND] + HP[TIED][BEHIND] / 2 + HP[AHEAD][TIED] / 2                #negative potential NNPOT callcs
    Npotdenominator = sum(HP[AHEAD]) + sum(HP[TIED])
    Npot = max(0.0, min(1.0, NPOTNumerator / Npotdenominator if Npotdenominator > 0 else 0.0))
 

    return max(0.0, min(1.0, HS * (1 - Npot) + (1 - HS) * Ppot))             #final calc, kept betweenn 1 and 0
 
 
 
def PreflopBucketMaker(Bucketsamount, Simamount, SaveFile='preflop_buckets.pkl'):          #Section for creating the preflop buckets

    print("Precomputinng Buckets\n")
    start = time.time()
    deck = [s * 100 + r for s in range(4) for r in range(2, 15)]            #create deck and then get all combinations (1326)
    Allhands = list(combinations(deck, 2))
 
    Handtypes = {}
    for hand in Allhands:                               #Groups hands by different types to cut down 1326 to 169, because 7c2s is the same as 7h2s
        cat = Hand_catgory(hand[0], hand[1])
        Handtypes.setdefault(cat, []).append(hand)
 

    categoryValue = {}                                            #Find equity for categories
    for idx, (cat, hands) in enumerate(Handtypes.items()):
        if idx % 10 == 0:
            print(f"  {idx}/{len(Handtypes)} categories ({100*idx/len(Handtypes):.0f}%)")             #progress bar printer
        rep = hands[0]
        categoryValue[cat] = PreflopStrenngthCalc(rep[0], rep[1], Simamount)
 

    Sortedvalues = sorted(categoryValue.items(), key=lambda x: x[1])                          #sorts the categories by their valiue and places into equal freuqnncy buckets
    bucketitemNumber = len(Sortedvalues) / Bucketsamount
    Bucketsorting = {cat: min(int(i / bucketitemNumber), Bucketsamount - 1)
                        for i, (cat, _) in enumerate(Sortedvalues)}
 

    PreflopBuckets = {}
    for cat, hands in Handtypes.items():                #Expands the categoires back into thr hands in those categories
        b = Bucketsorting[cat]
        for hand in hands:
            PreflopBuckets[tuple(sorted(hand))] = b
 


    with open(SaveFile, 'wb') as f:                     #savves the file
        pickle.dump(PreflopBuckets, f)
    print("Finished precomputing buckets\n")
    return PreflopBuckets
 


 
def PreflopBucketFileLoader(LoadFile='preflop_buckets.pkl'):            #For loadinng the file that was precomputed fi there is one
    with open(LoadFile, 'rb') as f:
        return pickle.load(f)
 
 
 
def EHSbucket(ourcards, boardcards, buckets=20, PreflopBuckets=None, EHS_cache=None):           #Gets the bucket based on EHS uses precompouted lookup table for preflop and for postflop there is a cache that stores after calculation

    if len(boardcards) == 0 and PreflopBuckets is not None:                  #Preflop from precomputed file
        key = tuple(sorted(ourcards))
        if key in PreflopBuckets:
            return PreflopBuckets[key]
 
    if EHS_cache is not None:                                                  #Postflop looks in cache first
        cache_key = (tuple(sorted(ourcards)), tuple(sorted(boardcards)))
        if cache_key in EHS_cache:
            return EHS_cache[cache_key]
 
    ehs = EHSCalc(ourcards, boardcards, samples=100)                   #if not in cache compute the EHS
    bucket = min(int(ehs * buckets), buckets - 1)
 
    if EHS_cache is not None:
        EHS_cache[(tuple(sorted(ourcards)), tuple(sorted(boardcards)))] = bucket
 
    return bucket
 
 
class Node:                                                                #Class for the information set

    def __init__(self, InformationSet, NumActions):
        self.info_set = InformationSet                                        #inititlisation of required variables
        self.num_actions = NumActions
        self.regret_sum = np.zeros(NumActions, dtype=np.float32)
        self.strategy = np.zeros(NumActions, dtype=np.float32)
        self.strategy_sum = np.zeros(NumActions, dtype=np.float32)
 
    def GetStrategy(self, weight):                                  #gets probability of action based on total regret, if all regret zero fall back to uniform

        normalize = 0

        for a in range(self.num_actions):                                          #negative regrets are clmaped to zero
            self.strategy[a] = max(self.regret_sum[a], 0)
            normalize += self.strategy[a]



        for a in range(self.num_actions):                          #normalize prob distribution
            if normalize > 0:
                self.strategy[a] /= normalize
            else:                                                        #negative play uniform

                self.strategy[a] = 1.0 / self.num_actions

            self.strategy_sum[a] += weight * self.strategy[a]               #accumulate weighted strat
        return self.strategy.copy()
 
    def Averagestrat(self):                                          #get average strategy across all CFR iterations


        avg = np.zeros(self.num_actions)
        total = np.sum(self.strategy_sum)
        for a in range(self.num_actions):
            avg[a] = self.strategy_sum[a] / total if total > 0 else 1.0 / self.num_actions
        return avg
 
 
_ML_TO_CFR = {'fold': 0, 'call': 1, 'check': 1, 'raise': 2, 'bet': 2, 'showdown': 1}
 
 
def Card_indexer(card):                                #converts card into an index for encoding

    return (card // 100) * 13 + (card % 100 - 2)                   
 
 
def FeatureVector(holecards, board, street, SB,
                              pot, to_call, Betamount, seq_pos,
                              dimensions=155):

    feat = np.zeros(dimensions, dtype=np.float32)

    feat[min(street, 3)] = 1.0                                          #encoding for street
    feat[4] = 1.0 if SB else 0.0
    feat[5] = pot / 200.0
    feat[6] = to_call / 200.0
    feat[7] = Betamount / 4.0
    feat[8] = seq_pos / 9.0
 
    if len(holecards) >= 1:                                            #one hot hole card encoding
        idx = Card_indexer(holecards[0])
        if 0 <= idx < 52:
            feat[10 + idx] = 1.0
    if len(holecards) >= 2:
        idx = Card_indexer(holecards[1])
        if 0 <= idx < 52:
            feat[62 + idx] = 1.0
 

    for bc in board[:5]:                            #board card encoding
        slot = Card_indexer(bc) % 41
        feat[114 + slot] = 1.0
 
    return feat
 
 
class MLOpponent:

    Fold = 0
    Check_or_call = 1
    Bet_or_raise = 2
 
    def __init__(self, model_path: str):                                    #initialisation for training model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"cant find model bro: {model_path}")         #tries to load model
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
 
        self.model = data['model']                                 
        self.encoder = data['action_encoder']
        self.classes = list(data['action_encoder'].classes_)
        self.feature_dim = data.get('feature_dim', 155)
 
    def get_action(self, hole_cards, board, history, street,
                   legalactions, SB, pot, to_call, bet_count):

        al = ''.join(c for c in history.split('/')[-1] if c.isalpha())         #position within street used for feature vector
        seq_pos = min(len(al), 9)
  
        feat = FeatureVector(                                           #build feature vector
            hole_cards, board, street, SB,
            pot, to_call, bet_count, seq_pos, self.feature_dim
        )

        probs = self.model.predict_proba(feat.reshape(1, -1))[0]
        action_scores = {self.Fold: 0.0, self.Check_or_call: 0.0, self.Bet_or_raise: 0.0}           #average output from 3 action space
        for cls, p in zip(self.classes, probs):
            cfr_idx = _ML_TO_CFR.get(str(cls).lower())
            if cfr_idx is not None:
                action_scores[cfr_idx] += p
 
        best, best_score = None, -1.0                #make sure only legal actions and pick highest/best one
        for a in legalactions:
            if action_scores[a] > best_score:
                best_score = action_scores[a]
                best = a
 
        return best if best is not None else random.choice(legalactions)         #in case all are negative
 
 


 
class TexasHoldemTrainer:
 
    FOLD = 0
    Check_or_call = 1
    Bet_or_raise = 2
    NumActions = 3
 
    def __init__(self, buckets=20, Stacksize=100,
                 preflop_buckets=None, opponent: MLOpponent = None):
        self.node_map = {}            
        self.SMALL_BLIND = 1
        self.BIG_BLIND = 2
        self.num_buckets = buckets
        self.starting_stack = Stacksize
        self.preflop_buckets = preflop_buckets
        self.ehs_cache = {}            
        self.opponent = opponent
 
        if self.opponent is None:
            raise ValueError("No model in rn bro")
 

    def train(self, iterations):                                #training fucntion that runs the training for however many iterations

        cards = [s * 100 + r for s in range(4) for r in range(2, 15)]
 

        CFRUtility = 0.0
        OppUtil = 0.0
 
        for i in range(iterations):
            if i % 100 == 0 and i > 0:
                self.ehs_cache.clear()                    #cache clearer so doesnt get too big
            if i % 500 == 0 and i > 0:
                gc.collect()
 
            random.shuffle(cards)
 
 
            if i % 2 == 0:                                #alternative big and small blind positions
                CFRHoldings = [cards[0], cards[1]]   
                OppHoldings = [cards[2], cards[3]]
                CFRSB = True
            else:
                OppHoldings = [cards[0], cards[1]]   
                CFRHoldings = [cards[2], cards[3]]
                CFRSB = False
 

            SBHoldings = CFRHoldings if CFRSB else OppHoldings
            BBHoldings = OppHoldings if CFRSB else CFRHoldings
 

            board = []
            remaining = cards[4:]
 

            util = self.cfr_chance_sampling(                    #util for one hand
                CFRHoldings=CFRHoldings,
                OppHolding=OppHoldings,
                SBHoldings=SBHoldings,
                BBHoldings=BBHoldings,
                board=board,
                deck=remaining,
                history='',
                street=0,
                CFRSB=CFRSB,
            )
 

            CFRUtility += util                                #calcs util
            OppUtil -= util 
 

            if (i + 1) % 100 == 0:                                #progress printer to make me aware how fast it is running/running at all
                AverageCFR  = CFRUtility / (i + 1)
                AverageOpp  = OppUtil / (i + 1)
                pct = 100 * (i + 1) / iterations
                mem_info = ''
 
                print(f"Iter {i+1:6d}/{iterations} ({pct:5.1f}%) | "
                      f"CFR: {AverageCFR:+.4f}BB | Opp: {AverageOpp:+.4f}BB | "
                      f"Nodes: {len(self.node_map):7d}{mem_info}")
 
        print(f"YOOOOOOOO TRAINING IS DONE")
        print(f"CFR avg utility : {CFRUtility / iterations:+.4f} BB/hand")
        print(f"Opp avg utility : {OppUtil / iterations:+.4f} BB/hand")
        print(f"Total nodes     : {len(self.node_map)}")
 

    def cfr_chance_sampling(self, CFRHoldings, OppHolding,
                             SBHoldings, BBHoldings,
                             board, deck, history, street,
                             CFRSB):


        if self.terminal(history, street):                           #tracks if terminal state has occured
            return self.Reward(SBHoldings, BBHoldings, board,
                                   history, street, CFRSB)
 
 
        if self.RoundDone(history, street):                           #tracks if current betting round is done
            if street < 3:

                NextStreet           = street + 1                                        #moves to next betting round and deals next cards
                Nextboard, Nextdeck = self.deal(board, deck, NextStreet)
                return self.cfr_chance_sampling(
                    CFRHoldings, OppHolding, SBHoldings, BBHoldings,
                    Nextboard, Nextdeck,
                    history + '/', NextStreet, CFRSB,
                )
            else:

                return self.Reward(SBHoldings, BBHoldings, board,                  #if on river then showdown happens
                                       history, street, CFRSB)



        al = ''.join(c for c in history.split('/')[-1] if c.isalpha())        #turn tracking
        SBTurn = (street == 0 and len(al) == 0) or (len(al) % 2 == 0)
        CFRturn = (SBTurn == CFRSB)
 
        legalactions = self.Legalcheck(history, street)
        if not legalactions:
            return self.Reward(SBHoldings, BBHoldings, board,
                                   history, street, CFRSB)
 

        if not CFRturn:                                 #get opponent action
            pot = self.PotSize(history)
            to_call = self.Calltrack(history, street)
            bet_cnt = al.count('b') + al.count('r')
 
            action = self.opponent.get_action(
                hole_cards = OppHolding,
                board = board,
                history = history,
                street = street,
                legalactions = legalactions,
                SB = SBTurn,
                pot = pot,
                to_call = to_call,
                bet_count = bet_cnt,
            )
 
            act_str = self.Stringconvert(action, history, street)
            next_history = history + act_str
 

            return -self.cfr_chance_sampling(                          #return util
                CFRHoldings, OppHolding, SBHoldings, BBHoldings,
                board, deck, next_history, street, CFRSB,
            )
 

        info_set = self.retrieveInfoSet(CFRHoldings, board, history)           #update regrets at node
        if info_set not in self.node_map:
            self.node_map[info_set] = Node(info_set, self.NumActions)
 
        node = self.node_map[info_set]
        strategy = node.get_strategy(1.0)
 

        legalprobs = np.array([strategy[a] for a in legalactions], dtype=np.float64)  #legal action restriction
        total = legalprobs.sum()
        if total <= 0:
            legalprobs = np.ones(len(legalactions)) / len(legalactions)
        else:
            legalprobs /= total
 

        sampled_action = int(np.random.choice(legalactions, p=legalprobs))                #sample only one action to traverse
        act_str = self.Stringconvert(sampled_action, history, street)
        next_history = history + act_str
 
        sampled_util = -self.cfr_chance_sampling(
            CFRHoldings, OppHolding, SBHoldings, BBHoldings,
            board, deck, next_history, street, CFRSB,
        )
 

        action_utils = {}                                       #recurses for no sampled legal actions
        for a in legalactions:
            if a == sampled_action:
                action_utils[a] = sampled_util
            else:
                alt_str = self.Stringconvert(a, history, street)
                alt_hist = history + alt_str
                action_utils[a] = -self.cfr_chance_sampling(
                    CFRHoldings, OppHolding, SBHoldings, BBHoldings,
                    board, deck, alt_hist, street, CFRSB,
                )
 

        for a in legalactions:                                              #update regrets
            node.regret_sum[a] += action_utils[a] - sampled_util
 
        return sampled_util
 

    def terminal(self, history, street):                                      #used for ternimal state checking


        current = history.split('/')[-1] if history else ''
        if 'f' in current:
            return True
        if street == 3 and self.RoundDone(history, street):
            return True
        return False
 
    def RoundDone(self, history, street):                                       #used for detecting betting round ending
        if not history:                               #goes through each possibility of round ending
            return False
        current = history.split('/')[-1]
        if not current:
            return False
        al = ''.join(c for c in current if c.isalpha())
        if len(al) < 2:
            return False
        if 'f' in al:
            return True
        last, second_last = al[-1], al[-2]
        if last == 'k' and second_last == 'k':
            return True
        if last == 'c' and second_last in ('b', 'r'):
            return True
        if (al.count('b') + al.count('r')) >= 4 and last == 'c':            #max action round end
            return True
        return False
 
    def Legalcheck(self, history, street):                                #determines legal actions at current point

        current = history.split('/')[-1] if history else ''
        al = ''.join(c for c in current if c.isalpha())
        aggression = al.count('b') + al.count('r')
        raisecheck = aggression < 4
 
        if not al:                                        #first action on street
            if street == 0:
                return [self.FOLD, self.Check_or_call, self.Bet_or_raise] if raisecheck else [self.FOLD, self.Check_or_call]
            else:
                return [self.Check_or_call, self.Bet_or_raise] if raisecheck else [self.Check_or_call]
 
        last = al[-1]
        if last == 'k':                                                                            #checls last action and then sees what is legal like
            return [self.Check_or_call, self.Bet_or_raise] if raisecheck else [self.Check_or_call]
        

        if last in ('b', 'r'):
            return [self.FOLD, self.Check_or_call, self.Bet_or_raise] if raisecheck else [self.FOLD, self.Check_or_call]
        
        if last == 'c':
            return [self.Check_or_call, self.Bet_or_raise] if raisecheck else [self.Check_or_call]
        

        if last == 'f':
            return []
        

        return [self.Check_or_call, self.Bet_or_raise]
 
    def Stringconvert(self, action, history, street):            #converts actions into string
        if action == self.FOLD:
            return 'f'
        current = history.split('/')[-1] if history else ''
        al = ''.join(c for c in current if c.isalpha())
        if action == self.Check_or_call:
            if not al:
                return 'c' if street == 0 else 'k'
            return 'c' if al[-1] in ('b', 'r') else 'k'
        if action == self.Bet_or_raise:
            if not al or al[-1] == 'k':
                return 'b'
            return 'r'
        return ''
 
    def deal(self, board, deck, street):                              #deals the card depending on street

        Newboard = board.copy()
        Newdeck = deck.copy()
        if street == 1:
            Newboard.extend(deck[:3]);  Newdeck = deck[3:]
        elif street in (2, 3):
            Newboard.append(deck[0]);   Newdeck = deck[1:]
        return Newboard, Newdeck
 
    def retrieveInfoSet(self, player_cards, board, history):           #gets info set string

        bucket = EHSbucket(
            player_cards, board, self.num_buckets,
            self.preflop_buckets, self.ehs_cache
        )
        return f"B{bucket}|{history}"
 
    def PotSize(self, history):                        #gets pot size based on actions in the history

        pot = self.SMALL_BLIND + self.BIG_BLIND
        streets = history.split('/')
        for idx, sa in enumerate(streets):
            bet_size = self.BIG_BLIND if idx <= 1 else 2 * self.BIG_BLIND
            for act in sa:
                if act in ('b', 'r', 'c'):
                    pot += bet_size
        return pot
 
    def Calltrack(self, history, street):                           #how much player needs to put in the match a bet

        current = history.split('/')[-1] if history else ''
        al = ''.join(c for c in current if c.isalpha())
        bet_size = self.BIG_BLIND if street <= 1 else 2 * self.BIG_BLIND
        return bet_size if al and al[-1] in ('b', 'r') else 0
 
    def Reward(self, SBHoldings, BBHoldings, board, history, street, CFRSB):                #finds reward for winning hand

        pot = self.PotSize(history)
        current = history.split('/')[-1] if history else ''
        al = ''.join(c for c in current if c.isalpha())
 

        if 'f' in al:                                       #reward from opponent fold
            actions_before_fold = al.split('f')[0]
            bb_folded = (len(actions_before_fold) % 2 == 0)
            SBwin = not bb_folded
            if CFRSB:
                return (pot / 2) if SBwin else -(pot / 2)
            else:
                return -(pot / 2) if SBwin else (pot / 2)
 
        if len(board) >= 5:                               #reward from showdown
            SBstrength = HandRanker(SBHoldings, board[:5])
            BBstrength = HandRanker(BBHoldings, board[:5])
            if SBstrength == BBstrength:
                return 0.0
            SBwin = SBstrength > BBstrength
            if CFRSB:
                return (pot / 2) if SBwin else -(pot / 2)
            else:
                return -(pot / 2) if SBwin else (pot / 2)
 
        return 0.0
 

    def Save(self, filename='cfr_vs_opponent_model.pkl'):                     #save model when done

        data = {
            'node_map': self.node_map,
            'num_buckets': self.num_buckets,
            'starting_stack': self.starting_stack,
            'small_blind': self.SMALL_BLIND,
            'big_blind': self.BIG_BLIND,
            'preflop_buckets': self.preflop_buckets,
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        size_mb = os.path.getsize(filename) / 1024 / 1024
        print(f"Model saved → {filename}  ({size_mb:.2f} MB, {len(self.node_map)} nodes)")
 
def main():           #main

    OPPONENT_MODEL_FILE = 'model_cluster_1.pkl'               #configuration area
    PREFLOP_BUCKET_FILE = 'preflop_buckets_10.pkl'     
    OUTPUT_MODEL_FILE   = 'cfr_vs_opponent_model1.pkl' 
    ITERATIONS          = 1_000_000                     
    NUM_BUCKETS         = 10                           
    STACK_SIZE          = 100                          
 

    if os.path.exists(PREFLOP_BUCKET_FILE):                                 #calc preflop bucket
        preflop_buckets = PreflopBucketFileLoader(PREFLOP_BUCKET_FILE)
    else:
        preflop_buckets = PreflopBucketMaker(
            Bucketsamount=NUM_BUCKETS,
            Simamount=100_000,
            SaveFile=PREFLOP_BUCKET_FILE,
        )
 
    opponent = MLOpponent(OPPONENT_MODEL_FILE)                      #gets CFR opponent

    trainer = TexasHoldemTrainer(                              #train CFR
        buckets = NUM_BUCKETS,
        Stacksize = STACK_SIZE,
        preflop_buckets = preflop_buckets,
        opponent = opponent,
    )
    trainer.train(ITERATIONS)
 
    trainer.Save(OUTPUT_MODEL_FILE)                                 #save the output model
  
 
if __name__ == '__main__':
    main()
