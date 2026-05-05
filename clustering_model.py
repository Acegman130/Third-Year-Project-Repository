import os
import pickle
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


try:                                                    #looks to find plotting, if cant then disables
    from plotting import ClusterPlotter
    PLOTTING_ENABLED = True
except ImportError:
    print("Warning: plotting.py not found. Plotting disabled.")
    PLOTTING_ENABLED = False


class CardEncoder:                 #class for card encoding

    cardvalues = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    }
    suits = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

    @staticmethod
    def CardFeatures(card: str) -> List[float]:                    #turns the rank and suit of a card into features for a feature vector
                                                                    #card value is normalised, four dimensions used to represent suit, five in total for a card

        if len(card) < 2:
            return [0.0] * 5

        values = CardEncoder.cardvalues.get(card[0], 0) / 14.0      #normalize by max value 
        featuresuits = [0.0, 0.0, 0.0, 0.0]                           #set to 1 for the suit it is
        suit_idx = CardEncoder.suits.get(card[1])
        if suit_idx is not None:
            featuresuits[suit_idx] = 1.0
        return [values] + featuresuits

    @staticmethod
    def StringtoFeature(cardstring: str, maxcards: int = 2) -> List[float]:                  #converts the card history string into a feautre vector by concatenating indivudal card vectors

        if not cardstring or len(cardstring) < 2:
            return [0.0] * (maxcards * 5)

        features: List[float] = []
        n = 0
        i = 0

        while i + 1 < len(cardstring) and n < maxcards:
            features.extend(CardEncoder.CardFeatures(cardstring[i:i + 2]))
            n += 1
            i += 2


        while n < maxcards:                           #zero pad cards features that are not present yet e.g. only flop
            features.extend([0.0] * 5)
            n += 1

        return features



Actions = ['fold', 'call', 'raise', 'check', 'bet']


def ActionPosition(sequence: List[Dict]) -> List[int]:                       #gets indice of every action in sequence

    return [
        i for i, a in enumerate(sequence)
        if a.get('type') == 'action' and a.get('player') == 'p1'
    ]


def Cleaner(sequence: List[Dict]) -> List[Dict]:                                         #remeoves any empty/None actions in the sequence
    return [a for a in sequence if a is not None]


class SupervisedPokerModel:


    def __init__(self, max_sequence_length: int = 10):
        self.max_sequence_length = max_sequence_length

        self.action_encoder = LabelEncoder()

        self.feature_dim = None
        self.is_trained = False

        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )


    def GetSequenceFeatures(self, sequence: List[Dict]) -> np.ndarray:                  #converts a sequence into a feature vector
 
        sequence = Cleaner(sequence)

        Holecards = ''      
        Communitycards = ''    
        P1actions: List[str] = []   
        P2actions: List[str] = []   

        for action in sequence:
            t = action.get('type')
            if t == 'deal' and action.get('player') == 'p1':
                Holecards = action.get('cards', '')
            elif t == 'board':
                Communitycards += action.get('cards', '')
            elif t == 'action':
                if action.get('player') == 'p1':
                    P1actions.append(action.get('action'))
                elif action.get('player') == 'p2':
                    P2actions.append(action.get('action'))

        features: List[float] = []


        features.extend(CardEncoder.StringtoFeature(Holecards, maxcards=2))                   #encodes hole cards


        features.extend(CardEncoder.StringtoFeature(Communitycards, maxcards=5))        #encode board cards

        for actions in (P1actions, P2actions):                                   #encodes the most recent actions of P1 and P2
            recent = actions[-self.max_sequence_length:]

            for i in range(self.max_sequence_length):
                if i < len(recent):
                    a = recent[-(i + 1)]  

                    features.extend(1.0 if a == act else 0.0
                                    for act in Actions)
                else:
                    features.extend([0.0] * len(Actions))

        return np.array(features)


    def train(self, sequences: List[List[Dict]]):                                 #train model to predict actions based on sequnces

        x, y = [], []
        for sequence in sequences:
            sequence = Cleaner(sequence)
            indices = ActionPosition(sequence)

            for i in range(len(indices) - 1):
                cur, nxt = indices[i], indices[i + 1]
                x.append(self.GetSequenceFeatures(sequence[:cur + 1]))

                y.append(sequence[nxt]['action'])

        if not x:
            self.is_trained = False
            return

        x = np.array(x)
        if self.feature_dim is None:
            self.feature_dim = x.shape[1]


        self.action_encoder.fit(y)
        self.model.fit(x, self.action_encoder.transform(y))
        self.is_trained = True

    def predict(self, context: List[Dict]) -> str:             #predict action given context

        if not self.is_trained:
            return random.choice(Actions)

        features = self.GetSequenceFeatures(context)
        if self.feature_dim and len(features) != self.feature_dim:
            return random.choice(Actions)

        try:
            pred = self.model.predict(features.reshape(1, -1))[0]

            return self.action_encoder.inverse_transform([pred])[0]
        except Exception:
            return random.choice(Actions)

    def AccuracyCalc(self, sequence: List[Dict]) -> Tuple[int, int]:          #calcs accuracy of action prediction based on how many it got right over the total

        if not self.is_trained:
            return 0, 0

        sequence = Cleaner(sequence)
        indices = ActionPosition(sequence)
        if len(indices) <= 1:
            return 0, 0

        correct = 0
        for i in range(len(indices) - 1):
            cur, nxt = indices[i], indices[i + 1]

            if self.predict(sequence[:cur + 1]) == sequence[nxt]['action']:
                correct += 1
        return correct, len(indices) - 1


    def save(self, filepath: str):               #saves models as output

        with open(filepath, 'wb') as f:
            pickle.dump({
                'model':               self.model,
                'action_encoder':      self.action_encoder,
                'max_sequence_length': self.max_sequence_length,
                'feature_dim':         self.feature_dim,
                'is_trained':          self.is_trained,
            }, f)
        print(f"Model saved to {filepath}")


class KModelsClustering:


    def __init__(self, k: int,
                 Maxiterations: int = 100,
                 Plotting: bool = True,
                 Plotoutput: str = './cluster_plots',
                 CheckpointOutput: str = './checkpoints',
                 CheckpointInterval: int = 5):
        
        self.k = k
        self.max_iterations = Maxiterations
        self.enable_plotting = Plotting and PLOTTING_ENABLED
        self.plot_output_dir = Plotoutput
        self.checkpoint_dir = CheckpointOutput
        self.checkpoint_interval = CheckpointInterval

        self.clusters: Dict[int, int] = {}
        self.models: List[SupervisedPokerModel] = []
        self.sequences: Dict[int, List[Dict]] = {}
        self.current_iteration = 0

        if self.enable_plotting:
            os.makedirs(Plotoutput, exist_ok=True)
        os.makedirs(CheckpointOutput, exist_ok=True)


    def ModelTrainer(self, cluster_id: int) -> SupervisedPokerModel:                 #trains the supervised models on sequences in that cluster

        ClustersSequnces = [
            self.sequences[sid]
            for sid, cid in self.clusters.items()
            if cid == cluster_id
        ]
        model = SupervisedPokerModel(max_sequence_length=10)
        model.train(ClustersSequnces)
        return model

    @staticmethod
    def ActionsInCluster(cluster_seqs: List[List[Dict]]) -> List[str]:         #gets list of every action in a cluster

        return [
            a.get('action')
            for seq in cluster_seqs
            for a in seq
            if a and a.get('type') == 'action' and a.get('player') == 'p1'
        ]

    def _get_cluster_sequences(self, cluster_id: int) -> List[List[Dict]]:            #returns all sequence IDs in a cluster
        return [
            self.sequences[sid]
            for sid, cid in self.clusters.items()
            if cid == cluster_id
        ]

    def ClusterSize(self):                                      #shows size of clusters
        counts = Counter(self.clusters.values())
        for cid in range(self.k):
            print(f"  Cluster {cid}: {counts.get(cid, 0)} sequences")

    def ClusterComposition(self):                                                  #hget the composition of actions in a cluster
        for cid in range(self.k):
            actions = self.ActionsInCluster(self._get_cluster_sequences(cid))
            if not actions:
                continue
            counts = Counter(actions)
            total = len(actions)
            top = ' '.join(f"{a}={100 * c / total:.0f}%"
                           for a, c in counts.most_common(3))
            print(f"  Cluster {cid} actions: {top} ")

    def Clusterplot(self, iteration: int):                                          #generate plots of results for each clsuter

        if not self.enable_plotting:
            return
        path = f"{self.plot_output_dir}/clusters_iteration_{iteration}.png"
        print(f"\nGenerating cluster plot for iteration {iteration}...")
        ClusterPlotter.plot_clusters(self.sequences, self.clusters,
                                     iteration=iteration, output_path=path)

    # ----- Checkpointing --------------------------------------------------

    def CheckpointSave(self, iteration: int):                   #Saves checkpoint of progress as cannot run all in one go

        data = {
            'iteration':      iteration,
            'clusters':       self.clusters,
            'models':         self.models,
            'k':              self.k,
            'max_iterations': self.max_iterations,
            'sequences':      self.sequences,
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_iter_{iteration}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        with open(os.path.join(self.checkpoint_dir, 'checkpoint_latest.pkl'), 'wb') as f:
            pickle.dump(data, f)
        print(f"\n💾 Checkpoint saved: {path}")

    @classmethod
    def LoadCheckpoint(cls, checkpoint_path: str):              #loads the checkpoints if there are any

        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        instance = cls(
            k=data['k'],
            max_iterations=data['max_iterations'],
        )
        instance.current_iteration = data['iteration']
        instance.clusters = data['clusters']
        instance.models = data['models']
        instance.sequences = data['sequences']
        print(f"\n📂 Checkpoint loaded from iteration {instance.current_iteration}")
        print(f"   Resuming from iteration {instance.current_iteration + 1}")
        return instance


    def fit(self, sequences: Dict[int, List[Dict]],                          #runs the k models clustering on the hand sequences in each cluster
            resume: bool = False) -> Dict[int, int]:

        self.sequences = sequences
        sequence_ids = list(sequences.keys())


        iteration = 0
        latest_checkpoint = os.path.join(self.checkpoint_dir,
                                         'checkpoint_latest.pkl')

        if resume and os.path.exists(latest_checkpoint):
            print(f"\n🔄 RESUMING from checkpoint...")
            loaded = KModelsClustering.LoadCheckpoint(latest_checkpoint)
            self.clusters = loaded.clusters
            self.models = loaded.models
            self.current_iteration = loaded.current_iteration
            iteration = self.current_iteration
        else:
            if resume:
                print(f"\n⚠️  No checkpoint found at {latest_checkpoint}")
                print(f"    Starting fresh...")

            print("\n" + "=" * 60)
            print("INITIALIZATION")
            print("=" * 60)


            print("Using random initialization")                         #on initialisation shuffles all sequnce ids into random clusters
            shuffled_ids = sequence_ids.copy()
            random.shuffle(shuffled_ids)
            for idx, seq_id in enumerate(shuffled_ids):
                self.clusters[seq_id] = idx % self.k

            print("Initial cluster assignment:")
            self.ClusterSize()
            self.ClusterComposition()
            self.Clusterplot(0)

        changed = True                                                  #main loop to train models and then reassign
        while changed and iteration < self.max_iterations:
            iteration += 1
            print(f"\n{'='*60}\nITERATION {iteration}\n{'='*60}")
            changed = False

            print("\nTraining models...")
            self.models = []
            for i in range(self.k):
                print(f"  Training model for cluster {i}...")
                self.models.append(self.ModelTrainer(i))


            print("\n" + "-" * 60)
            print("Reassigning sequences based on model accuracy on P1 actions...")
            print("-" * 60)

            Chnagedcount = 0
            print_interval = max(1, len(sequence_ids) // 10)

            for idx, seq_id in enumerate(sequence_ids):
                if idx and idx % print_interval == 0:
                    pct = 100 * idx / len(sequence_ids)
                    print(f"  Progress: {pct:.0f}% "
                          f"({idx}/{len(sequence_ids)} sequences)")

                sequence = self.sequences[seq_id]

                accuracies = []
                for model in self.models:
                    if model.is_trained:
                        c, t = model.AccuracyCalc(sequence)
                        accuracies.append(c / t if t > 0 else 0)
                    else:
                        accuracies.append(0)

                Highestaccuracy = max(accuracies)
                bestcluster = [i for i, a in enumerate(accuracies)
                                 if a == Highestaccuracy]
                current = self.clusters[seq_id]

                if current in bestcluster:
                    newcluster = current
                else:

                    trainedbest = [i for i in bestcluster
                                    if self.models[i].is_trained]
                    newcluster = random.choice(trainedbest or bestcluster)

                if newcluster != current:
                    self.clusters[seq_id] = newcluster
                    Chnagedcount += 1
                    changed = True

            print(f"\nReassignments: {Chnagedcount} sequences moved")
            self.ClusterSize()
            self.ClusterComposition()
            self.Clusterplot(iteration)

            if iteration % self.checkpoint_interval == 0:
                self.CheckpointSave(iteration)

        if iteration >= self.max_iterations:
            print(f"\nReached maximum iterations ({self.max_iterations})")
        else:
            print(f"\nConverged after {iteration} iterations")

        self.CheckpointSave(iteration)
        return self.clusters


    def SaveClusterModels(self, output_dir: str):                       #saves the clsuter models

        os.makedirs(output_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            with open(os.path.join(output_dir, f'cluster_{i}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
        with open(os.path.join(output_dir, 'clusters.pkl'), 'wb') as f:
            pickle.dump(self.clusters, f)
        print(f"Models and clusters saved to {output_dir}")

    def ClusterStats(self) -> Dict:                               #gets the stats for each cluster

        stats = {}
        for cid in range(self.k):
            cluster_seqs = self._get_cluster_sequences(cid)
            actions = self.ActionsInCluster(cluster_seqs)


            Totalcorrect = totalpredictions = 0
            model = self.models[cid] if cid < len(self.models) else None
            if model is not None:
                for seq in cluster_seqs:
                    c, t = model.AccuracyCalc(seq)
                    Totalcorrect += c
                    totalpredictions += t

            stats[cid] = {
                'num_sequences':  len(cluster_seqs),
                'actions_learned': Counter(actions),
                'avg_accuracy':   (Totalcorrect / totalpredictions
                                   if totalpredictions else 0),
            }
        return stats
