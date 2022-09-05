Repository for the paper "The Effectiveness of Masked Language Modeling and Adapters for Factual Knowledge Injection".

# Data processing

See the README at the data folder.

# Adapter training

The first step of the experiment is to train the adapters on a subgraph of ConceptNet. In 'run_mlm.sh', specify the PLM you want to use as the base language model together with the hyperparameters for the adaper and a path to the extracted data file. After the training completes, you will have a pytorch model file that contains the PLM with the additional adapter weights.

# Evaluation

In order to evaluate the adapter-injected models on the LAMA probe, specify a path to the injected model in run_lama_probe.sh and set the "--use\_adapter " flag. You can specify what predicate types you want to limit the probe to.
