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

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    """
    Parse the arguments from sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="model checkpoint")
    parser.add_argument("--dataset", type=str, help="dataset")
    parser.add_argument("--prepared_dataset_dir", type=str, help="transformed dataset")
    parser.add_argument("--output_dir", type=str, help="output dir")

    return parser.parse_args()


prefix = "summarize: "


def preprocess(examples, tokenizer):
    inputs = [prefix + doc for doc in examples["report"]]

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


if __name__ == "__main__":
    args = parse_args()

    ds = load_dataset(args.dataset)
    tokennizer = AutoTokenizer.from_pretrained(args.checkpoint)

    tokenized_dataset = ds.map(
        lambda examples: preprocess(examples, tokennizer), batched=True
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["report", "summary"])

    tokenized_dataset.save_to_disk(args.prepared_dataset_dir)
