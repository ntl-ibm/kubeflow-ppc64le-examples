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
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
import os


def parse_args() -> argparse.Namespace:
    """
    Parse the arguments from sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="model checkpoint")
    parser.add_argument(
        "--split",
        type=str,
        help="split of billsum dataset",
        choices=["ca_test", "train", "test"],
        default="ca_test",
    )
    parser.add_argument("--prepared_dataset_dir", type=str, help="transformed dataset")
    parser.add_argument("--prefix", type=str, help="prefix", default="summarize: ")
    parser.add_argument(
        "--model_max_len",
        type=int,
        default=512,
        help="max number of tokens the model can handel",
    )

    return parser.parse_args()


prefix = "summarize: "


def preprocess(
    examples, tokenizer, model_max_len
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=model_max_len, truncation=True)

    labels = tokenizer(
        text_target=examples["summary"], max_length=model_max_len, truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    args = parse_args()

    billsum = load_dataset("billsum", split=args.split)

    train_valid_test = billsum.train_test_split(test_size=0.2)
    train_valid = train_valid_test["train"].train_test_split(test_size=0.2)

    billsum = DatasetDict(
        {
            "train": train_valid["train"],
            "valid": train_valid["test"],
            "test": train_valid_test["test"],
        }
    )

    print(billsum)
    tokennizer = AutoTokenizer.from_pretrained(args.checkpoint)

    tokenized_dataset = billsum.map(
        lambda examples: preprocess(examples, tokennizer, args.model_max_len),
        batched=True,
    )

    os.makedirs(args.prepared_dataset_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(args.prepared_dataset_dir)
