# Copyright 2023 IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Script to fine tune an LLM
"""
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os


from metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    """
    Parse the arguments from sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="model checkpoint")
    parser.add_argument("--prepared_dataset_dir", type=str, help="transformed dataset")
    parser.add_argument("--model_dir", type=str, help="output model")
    parser.add_argument("--epochs", type=int, help="epochs", default=3)
    parser.add_argument("--optim", type=str, help="optimizer", default="adafactor")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--tensorboard", type=str, help="tensorboard", default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.checkpoint)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./model",
        evaluation_strategy="epoch",
        logging_steps=int(args.batch_size / 2),
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        # Used to reduce memory, with the price tag that training is much slower
        gradient_checkpointing=True,
        optim=args.optim,
        logging_dir=args.tensorboard,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    tokenized_dataset = load_from_disk(args.prepared_dataset_dir)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    trainer.train()

    os.makedirs(args.model_dir, exist_ok=True)
    trainer.save_model(output_dir=args.model_dir)
