import random
import numpy as np
from clustering_model import KModelsClustering
from data_pipeline import Completebuild


FOLDER_PATH = "drive/MyDrive/VaryHands"                 #Config Area
PLAYER_NAME = None                        
POSITION = 'p1'                          
MAX_HANDS = 250000                         


N_CLUSTERS = 3                            
MAX_ITERATIONS = 20                       


OUTPUT_DIR = './models'                   
PLOT_DIR = './cluster_plots'               
CHECKPOINT_DIR = './checkpoints'           
CHECKPOINT_INTERVAL = 5                  


RESUME = False                             


def main():

    random.seed(42)
    np.random.seed(42)


    print("=" * 80)
    print("STEP 1: PARSING DATA AND EXTRACTING SEQUENCES")
    print("=" * 80)

    result = Completebuild(
        folder=FOLDER_PATH,
        player=PLAYER_NAME,
        position=POSITION,
        clusternumber=N_CLUSTERS,
        maxhands=MAX_HANDS,
        maxiterations=MAX_ITERATIONS,
        output=OUTPUT_DIR,
        random_seed=42,
    )
    handsequences = result['handsequences']
    print(f"\n✓ Ready to cluster {len(handsequences)} hand sequences")


    print("\n" + "=" * 80)
    print("STEP 2: RUNNING K-MODELS CLUSTERING WITH VISUALIZATION")
    print("=" * 80)

    clustering = KModelsClustering(
        k=N_CLUSTERS,
        Maxiterations=MAX_ITERATIONS,
        Plotting=True,
        Plotoutput=PLOT_DIR,
        CheckpointOutput=CHECKPOINT_DIR,
        CheckpointInterval=CHECKPOINT_INTERVAL,
    )

    if RESUME:
        print(f"\n🔄 Attempting to resume from {CHECKPOINT_DIR}/checkpoint_latest.pkl")
    else:
        print(f"\n▶️  Starting fresh (resume=False)")
    clustering.fit(hand_sequences, resume=RESUME)


    print("\n" + "=" * 80)
    print("STEP 3: CLUSTER STATISTICS")
    print("=" * 80)

    for cluster_id, cs in clustering.ClusterStats().items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Number of hands: {cs['num_sequences']}")
        print(f"  Average accuracy: {cs['avg_accuracy']:.3f}")

        actions = cs['actions_learned']
        total = sum(actions.values())
        print(f"  Top actions:")
        for action, count in actions.most_common(5):
            pct = 100 * count / total if total else 0
            print(f"    {action}: {count} ({pct:.1f}%)")


    print("\n" + "=" * 80)
    print("STEP 4: SAVING MODELS")
    print("=" * 80)
    clustering.SaveClusterModels(OUTPUT_DIR)
    print(f"\n✓ Models saved to: {OUTPUT_DIR}")
    print(f"✓ Plots saved to: {PLOT_DIR}")
    print(f"✓ Checkpoints saved to: {CHECKPOINT_DIR}")


    print("\n" + "=" * 80)
    print("STEP 5: TESTING PREDICTIONS")
    print("=" * 80)

    test_aggressive = [
        {'type': 'deal',   'player': 'p1', 'cards': 'AsKs'},
        {'type': 'deal',   'player': 'p2', 'cards': 'QdJc'},
        {'type': 'action', 'player': 'p2', 'action': 'raise', 'amount': 4},
        {'type': 'action', 'player': 'p1', 'action': 'raise', 'amount': 10},
        {'type': 'board',  'cards': 'AcKdQh'},
        {'type': 'action', 'player': 'p2', 'action': 'call',  'amount': 0},
    ]

    test_passive = [
        {'type': 'deal',   'player': 'p1', 'cards': '7s2d'},
        {'type': 'deal',   'player': 'p2', 'cards': 'AhKh'},
        {'type': 'action', 'player': 'p2', 'action': 'raise', 'amount': 4},
    ]

    def show_predictions(label, header, sequence):

        print(f"\n{label}")
        print(header)
        for i, model in enumerate(clustering.models):
            pred = model.predict(sequence) if model.is_trained else "(not trained)"
            print(f"  Cluster {i}: {pred}")

    show_predictions(
        "Test sequence: Aggressive play with strong cards (AsKs)",
        "Board: AcKdQh\nP1 actions so far: [raise 10]\n\n"
        "Predicted next P1 action by each cluster model:",
        test_aggressive,
    )
    show_predictions(
        "Test sequence: Weak cards (7s2d), opponent raises",
        "Predicted next P1 action by each cluster model:",
        test_passive,
    )


    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  📁 Models:      {OUTPUT_DIR}/  (cluster_*_model.pkl, clusters.pkl)")
    print(f"  📊 Plots:       {PLOT_DIR}/    (clusters_iteration_*.png)")
    print(f"  💾 Checkpoints: {CHECKPOINT_DIR}/  (checkpoint_iter_*.pkl, "
          f"checkpoint_latest.pkl)")
    print("\nYou can now:")
    print("  1. View the plots to see VPIP/PFR cluster evolution")
    print("  2. Load models to make predictions on new hands")
    print("  3. Analyze which playing styles were discovered")
    print("  4. Resume training if interrupted (set RESUME=True)")
    print("=" * 80)


if __name__ == "__main__":
    main()
