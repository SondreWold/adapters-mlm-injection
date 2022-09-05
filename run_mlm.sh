TRAIN_FILE="./data/concept_net/lama_corpus_train.txt"
VAL_FILE="./data/concept_net/lama_corpus_val.txt"
TOKENIZER="bert-base-uncased"
MODEL_TYPE="bert-base-uncased"
TRAINING_FOLDER="."
OUTPUT_DIR="./models/first_six_dropped"

python $TRAINING_FOLDER/run_mlm.py \
    --model_name_or_path $MODEL_TYPE \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --output_dir $OUTPUT_DIR \
    --line_by_line True \
    --adapter_config "houlsby" \
    --non_linearity "gelu" \
    --reduction_factor 12      \
    --num_warmup_steps 10000 \
    --max_train_steps 100000 
