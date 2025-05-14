# Dataset
The datasets used in this work, which include both Petri net models and event logs, can be found in the [evaluation data repository](https://gitlab.com/dominiquesommers/thesis_data/-/tree/master?ref_type=heads) by [1]. You can download the following files:

    Petri net files (.pnml)

    Event log files (.xes)

Once you have the datasets, place the .pnml and .xes files in the data folder, and the script will use them for evaluation.

# Running the Script

We evaluate the similarity between the discovered process model and the ground truth by comparing the trace sets σ* and σ. In the static model approach, all possible sequences over the activity set 𝒜ₘ are considered, i.e., σ* ⊆ ℬₘ. Alternatively, in the context-aware approach, the discovered process model is refined based on the observed trace σ, leading to a more constrained comparison: σ* ⊆ |σ ∩ ℬₘ|.



To run the script, use the following command:

python main.py <mode>

Where <mode> is either:

    case: Process the event logs and generate case-based trace graphs. (Context-Aware Model)

    mined: Process Petri net models and generate mined trace graphs. (Static Model)

Example:

python main.py case

This will process the dataset in case mode and generate embeddings based on the case-based trace graphs.
Output

    The results will be saved in the results directory, including:

        PCA visualizations of the trace embeddings and their comparison with the truth embeddings.

        CSV files with computed distance metrics between the trace and truth embeddings.

## Functions

    main(mode): The main function that orchestrates the conformance checking process.

    compare_vectors_with_penalty: Compares two sets of Node2Vec embeddings using a penalty factor.
# Reference
    [1] . Sommers, V. Menkovski, and D. Fahland, “Supervised learning of process discovery techniques using graph neural networks,” Information Systems, vol. 115, p. 102209, 2023.
