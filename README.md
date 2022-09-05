# Adapter Injection and MLM

Code and data associated with the paper "The Effectiveness of Masked Language Modeling and Adapters for Factual Knowledge Injection"

### Data processing

See the README at the data folder.

### Adapter training

The first step of the experiment is to train the adapters on a subgraph of ConceptNet. In ´´´run_mlm.sh´´´, specify the PLM you want to use as the base language model together with the hyperparameters for the adaper and a path to the extracted data file. After the training completes, you will have a pytorch model file that contains the PLM with the additional adapter weights.

### Evaluation

In order to evaluate the adapter-injected models on the LAMA probe, specify a path to the injected model in run_lama_probe.sh and set the ´´´--use\_adapter ´´´ flag. You can specify what predicate types you want to limit the probe to.

### Requirements

´´´
networkx             - 2.5
adapter-transformers - 2.2.0
transformers         - 4.3.3
pytorch              - 1.7.1
python               - 3.7.7
´´´

