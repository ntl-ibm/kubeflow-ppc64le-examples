import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
from pathlib import Path
import json

from metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    """
    Parse the arguments from sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dataset_dir", type=str, help="transformed dataset")
    parser.add_argument("--model_dir", type=str, help="output model")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument(
        "--results_json", type=str, help="results file", default="./results.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model_dir)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    tokenized_dataset = load_dataset(args.prepared_dataset_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir="/tmp/predictions",
        per_device_eval_batch_size=args.batch_size,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    results = trainer.evaluate(tokenized_dataset["test"])

    os.makedirs(Path(args.results_json).parent.absolute(), exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(results, f)
