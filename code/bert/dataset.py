from copy import deepcopy
from dataclasses import dataclass

from datasets import Dataset
from transformers import AutoTokenizer


def get_tokenizer(config):
    return AutoTokenizer.from_pretrained(
        config.dataset.tokenizer_path,
        use_fast=config.dataset.use_fast,
        padding_side=config.dataset.padding_side,
        truncation_side=config.dataset.truncation_side,
    )


class IMDBDataset:
    """
    Dataset class for IMDB
    """

    def __init__(self, config):
        self.config = config
        self.tokenizer = get_tokenizer(config)

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=self.config.model.max_seq_len,
            add_special_tokens=True,
        )

        return tz

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def preprocess_function(self, df):
        df["text"] = df["text"].apply(
            lambda x: x.strip() + "\n###\nWhat is the sentiment?"
        )
        return df

    def get_dataset(self, df):
        df = deepcopy(df)
        df = self.preprocess_function(df)
        task_dataset = Dataset.from_pandas(df)

        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)

        task_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )

        if (
            "label" in task_dataset.column_names
            and "labels" not in task_dataset.column_names
        ):
            task_dataset = task_dataset.rename_column("label", "labels")

        return task_dataset
