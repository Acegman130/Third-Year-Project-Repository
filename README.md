Cluster-based Opponent Modelling and its Impact on Counterfactual Regret Minimisation Approaches to Imperfect Information Games

Introduction:
This software makes up the entirety of the work on this project. It is split into two main parts: the Counterfactual Regret Minimisation section (CFR) and the K-models clustering section. The file called CFR makes up all of the code for the CFR algorithm and was used to output all the CFR trained models. The rest of the code files are all related to the function of the K-models clustering algorithm which produced supervised learning models from each cluster to then be used in the CFR algorithm.

Contextual Overview:
The data flows through the system in the following order:
1.	Raw data (.phhs files) — The files from the Annual Computer Poker Competition are placed in the project folder ready to be parsed.
2.	data_parser.py — Parses the .phhs files and extracts all necessary data needed for creating the sequences used in the k models clustering algorithm.
3.	data_pipeline.py — Filters the parsed hands by player name and seat position and then converts the action strings into sequences ready for clustering.
4.	clustering_model.py — Runs the k-models clustering algorithm which finds clusters of hands that share similar player types and then trains a supervised learning models off the data in that cluster. These models are outputted for later
5.	plotting.py — Used to plot useful graphs that display different relevant statistics for each cluster. This is used to diagnose progress and see cluster development over iterations.
6.	Cluster models (cluster_*.pkl) — The trained supervised learning models from each cluster are saved.
7.	CFR.py — Each of the cluster saved models are then loaded one at a time against the CFR algorithm which then produces CFR models that are tailored to the clusters found by the k models algorithm.

Installation Instructions:

Prerequisites
•	Python 3.8 or later
•	pip (Python package manager)

Dependencies
Install all required packages such as:
• numpy	- Numerical arrays and linear algebra
• scikit-learn	- MLPClassifier and LabelEncoder for supervised models
• matplotlib - Cluster evolution plots
• treys -	Fast poker hand evaluation for EHS Squared calculation in plotting.py

Environment Setup
1.	Clone the repository from GitHub: https://github.com/Acegman130/Third-Year-Project-Repository.git
2.	Place the .phhs hand history files from the Annual Computer Poker Competition dataset in the project folder

How to Run the Software:
Stage 1: K-Models Clustering
Change the variables in the configuration section of the run_clustering.py file and the run that file.
Stage 2: CFR Counter-Strategy Training
Make sure the model you want to use as the opponent for the CFR is in your project folder and then edit the configuration settings as desired making sure the opponent filename is the same as the one you want. Then run CFR.py

Technical Details:
Most of the technical details necessary to understand the code are mentioned in the paper itself. The only one that was not is the Hand Ranker function. This function works is that it assigns a score to different hand categories based on its relative strength inside that category e.g. pair of aces will get 2900 while a three of a kind aces will get 4900. Each hand category gets a 1000 score range with the better hands in that category getting placed higher in the range.

Known Issues and Future Improvements:
Most of the known issues and future improvements are discussed in depth in the future work section of the paper however there are some that specifically related to the code which I will discuss:
•	Performance optimisation — The time take for the program to finish was a major issue across both the K-models clustering and CFR sections. Some speed ups could come from simple fixes such as creating array lookup tables in advance for some necessary sections. Other improvements may take the form of trying to parallelise bits of code that can be done in parallel. These improvements could also target memory issues as while my system always had enough RAM to run these programs, they were still incredibly RAM intensive. Without the extra RAM purchased through the university it would have been a struggle to run the files as they exist now.

Third Party/AI Involvement:
Through this project, Claude [1] has been used to help format/structure and debug code so has played a role in all of the files created. It is also responsible for the direct generation of any save/checkpointing functions along with the displaying of statistics such as the entirety of plotting.py among with functions in the other files. It is also responsible for the run_clustering.py file as this is just a file used to run all the other k models clustering files and track output.

[1] Anthropic Claude (2026).’Formatting and debugging of code along with the generation of statistics/display across the code’. Claude Opus 4. [AI assistant]. Available at: https://claude.ai/ (Accessed: 1 May 2026).
