MODEL="bert-base-uncased" #No trailing / !!
TOKENIZER="bert-base-uncased"

python ./lama_probe.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --gpu 0 \
  --relations IsA UsedFor AtLocation \ #Predicate types to test for. If not specified, all LAMA predicate types are included. 
  --full_eval
  #--use_adapter \
