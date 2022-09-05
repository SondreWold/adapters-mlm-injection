import argparse
import json
import logging
import random
import string
from tqdm import tqdm
from typing import List, Dict
from transformers import pipeline, Pipeline, AutoConfig, AutoTokenizer, AutoModelForMaskedLM
logger = logging.getLogger(__name__)
from datasets import load_dataset
import string
import numpy as np
from transformers.adapters.composition import Fuse


def evaluate_lama(model, data, at_k, relations=[], is_logging=False):
    '''
    Calculates the precision @ k for a model on the LAMA dataset. If k=1, then we get normal accuracy.
    Since we have only one relevant item irrespective of k, p@k=1 if the term is in the top k documents. 
    '''
    points = 0
    n = len(data)
    oov_words = 0
    logger.info(f"Relations specified: {relations}")
    for line in tqdm(data):
        if relations:
            if line["pred"] not in relations:
                n -= 1
                continue
        correct = line["obj_label"]
        # Ignore OOV words. 
        obj_label_id = model.tokenizer.vocab.get(correct)
        if obj_label_id is None:
            n -= 1
            oov_words += 1
            continue
        sentence = line["masked_sentence"].replace("[MASK]", "<mask>") if "roberta" in model.model.config.name_or_path else line["masked_sentence"]
        if is_logging:
            logger.info(f"Sentence is {sentence}")
            logger.info(f"Correct answer is {correct}")
        predictions = model(sentence)
        for pred in predictions:
            if is_logging: logger.info(f"Prediction was {pred['token_str']}")
            if pred["token_str"].strip().lower() == correct:
                points += 1
    if len(relations) == 1:
        logger.info(f"Relation was {relations[0]} with {n} samples, removed {oov_words} OOV words.")
    return points/n

def read_jsonl_file(filename: str) -> List[Dict]:
    dataset = []
    with open(filename) as f:
        for line in f:
            loaded_example = json.loads(line)
            dataset.append(loaded_example)

    return dataset

def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument(
        "--model_name_or_path", type=str, help="name of the used masked language model", default="bert-base-uncased")
    parse.add_argument("--gpu", type=int, default=-1)
    parse.add_argument("--lama_path", type=str, default=None)
    parse.add_argument("--at_k", type=int, default=5)
    parse.add_argument("--adapter_name", type=str, default=None)
    parse.add_argument("--adapter_fusion_path", type=str, default=None)
    parse.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parse.add_argument("--use_adapter", action='store_true')
    parse.add_argument('--full_eval', action='store_true')
    parse.add_argument('--use_fusion', action='store_true')
    parse.add_argument('--micro', action='store_true')
    parse.add_argument('--relations', nargs='*', default=[])
    parse.add_argument('--adapter_list', nargs='+', default=[], help="Path to Adapters to add to fusion layer")


    args = parse.parse_args()


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    data = load_dataset("lama", "conceptnet")["train"] #from Huggingface

    lm = args.model_name_or_path
    logging.info(f"Initializing a model from name or path: {lm} and tokenizer {args.tokenizer_name}")
    config = AutoConfig.from_pretrained(lm)
    base_model = AutoModelForMaskedLM.from_pretrained(lm, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    device = args.gpu

    name = lm if not args.use_adapter else args.adapter_name
    adapter_flag = "adapter" if args.use_adapter else "normal"
    if args.use_adapter:
        logger.info("Load Adapter model")
        if args.use_fusion:
            logger.info("Using AdapterFusion setup")
            adapters = args.adapter_list
            for adapter in adapters:
                logger.info(f"Loading adapter: {adapter}")
                base_model.load_adapter(adapter, with_head=False, set_active=True)
            logger.info(f"Loading Fusion layer from path: {args.adapter_fusion_path}")
            base_model.load_adapter_fusion(args.adapter_fusion_path, set_active=True)
        else:
            logger.info("ST-Adapter mode set.")
            base_model.set_active_adapters([args.adapter_name])
        base_model.freeze_model(False)

    if args.full_eval:
        results = {}
        for k in [1,10,100]:
            logging.info(f"Calculating for k={k}")
            model = pipeline("fill-mask", model=base_model,
                        tokenizer=tokenizer, device=device, top_k=k)
            mean_p_at_k = evaluate_lama(model, data, k, args.relations)
            logger.info(f"Precision for model @{k} was {mean_p_at_k}")
            results[k] = mean_p_at_k
        with open(f"./lama_results_{adapter_flag}_{name}_{args.tokenizer_name}_.txt", 'w+') as f:
            f.write(f"Results for model loaded from path {args.model_name_or_path} with tokenizer: {args.tokenizer_name} \n")
            for key, value in results.items():
                f.write(f"Precision@{key}: {value} \n")
                
    # Micro-averaged accuracy
    elif args.micro:
        logger.info(f"Calculating micro averages for k={args.at_k}")
        results = {}
        model = pipeline("fill-mask", model=base_model,
                        tokenizer=tokenizer, device=device, top_k=args.at_k)
        for relation in args.relations:
            accuracy = evaluate_lama(model, data, args.at_k, [relation])
            results[relation] = accuracy
        
        logger.info(results)
        logger.info(f"Micro-averaged accuracy for k {args.at_k}: {np.mean(list(results.values()))}")
    
    else:
        model = pipeline("fill-mask", model=base_model,
                        tokenizer=tokenizer, device=device, top_k=args.at_k)
        mean_p_at_k = evaluate_lama(model, data, args.at_k, args.relations)
        logger.info(f"Precision for model @{args.at_k} was {mean_p_at_k}")




if __name__ == '__main__':
    main()
