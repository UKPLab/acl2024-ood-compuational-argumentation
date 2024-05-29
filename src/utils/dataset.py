import pandas
import torch
from openprompt.data_utils import InputExample
from torch.utils.data import Dataset

from utils.composition import compose_samples

class InstructionDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.x = input_ids
        self.y = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        return {
            "input_ids": torch.tensor(self.x[index]),
            "label": torch.tensor(self.y[index])
        }

class SimpleDataset(Dataset):
    def __init__(self, file_name, task, text_encoding):
        self.samples = pandas.read_json(file_name)

        encoding_input = compose_samples(self.samples, task=task, sep_token=text_encoding.transforming.tokenizer.sep_token)

        self.x = list(text_encoding.encode(encoding_input))
        self.y = self.samples["label"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        return self.x[index],self.y[index]


class FineTuningDataset(Dataset):
    def __init__(self, samples):
        self.x = list(samples["input_ids"])
        self.y = list(samples["label"])

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        return {
            "input_ids": torch.tensor(self.x[index]),
            "label": torch.tensor(self.y[index])
        }


def convert_to_prompt_examples(samples):
    if "language" in samples.columns and "text" in samples.columns and "topic" in samples.columns:
        return [
            InputExample(
                guid=index,
                text_a=row["text"],
                text_b=row["topic"],
                label=row["label"],
                meta={"language": row["language"]}
            )
            for index, row in samples.iterrows()
        ]
    elif "topic" in samples.columns and "text" in samples.columns:
        return [
            InputExample(
                guid=index,
                text_a=row["text"],
                text_b=row["topic"],
                label=row["label"]
            )
            for index, row in samples.iterrows()
        ]
    elif "target" in samples.columns and "text" in samples.columns:
        return [
            InputExample(
                guid=index,
                text_a=row["text"],
                text_b=row["target"],
                label=row["label"]
            )
            for index, row in samples.iterrows()
        ]
    elif "language" in samples.columns and "premise" in samples.columns:
        return [
            InputExample(
                guid=index,
                text_a=row["premise"],
                text_b=row["hypothesis"],
                label=row["label"],
                meta={"language": row["language"]}
            )
            for index, row in samples.iterrows()
        ]
    elif "headline" in samples.columns and "text" in samples.columns:
        return [
            InputExample(
                guid=index,
                text_a=row["text"],
                text_b=row["headline"],
                label=row["label"]
            )
            for index, row in samples.iterrows()
        ]
    elif "text_1" in samples.columns and "text_2" in samples.columns and "topic" in samples.columns:
        return [
            InputExample(
                guid=index,
                text_a=row["text_1"],
                text_b=row["text_2"],
                label=row["label"],
                meta={"topic": row["topic"]}
            )
            for index, row in samples.iterrows()
        ]
    elif "text_1" in samples.columns and "text_2" in samples.columns :
        return [
            InputExample(
                guid=index,
                text_a=row["text_1"],
                text_b=row["text_2"],
                label=row["label"],
            )
            for index, row in samples.iterrows()
        ]
    elif "sentence1" in samples.columns and "sentence2" in samples.columns:
        return [
            InputExample(
                guid=index,
                text_a=row["sentence1"],
                text_b=row["sentence2"],
                label=row["label"],
            )
            for index, row in samples.iterrows()
        ]
    elif "text" in samples.columns and "domain" in samples.columns and "language" in samples.columns:
        return [
            InputExample(
                guid=index,
                text_a=row["text"],
                label=row["label"],
                meta={"language": row["language"]}
            )
            for index, row in samples.iterrows()
        ]
    elif "text" in samples.columns and "domain" in samples.columns:
        return [
            InputExample(
                guid=index,
                text_a=row["text"],
                label=row["label"],
            )
            for index, row in samples.iterrows()
        ]
    elif "text" in samples.columns and "year" in samples.columns:
        return [
            InputExample(
                guid=index,
                text_a=row["text"],
                label=row["label"],
            )
            for index, row in samples.iterrows()
        ]