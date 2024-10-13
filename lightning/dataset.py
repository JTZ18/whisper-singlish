"""
Create a PyTorch Custom dataset that loads file in data/other.tsv that contains
the path to image audio and text transcription.
"""

import pytorch_lightning as pl
from tqdm import tqdm
import ffmpeg
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
import sys


class IMDA(Dataset):
    def __init__(self, data_dir, whisper_model="tiny"):
        self.sampling_rate = 16_000
        self.data_dir = data_dir
        self.data = pd.read_parquet(
            os.path.join(data_dir, "data.parquet"),
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            f"openai/whisper-{whisper_model}"
        )
        self.tokenizer = WhisperTokenizer.from_pretrained(
            f"openai/whisper-{whisper_model}", language="en", task="transcribe"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_bytes = self.data.iloc[idx]["audio"]["bytes"]
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        sentence = self.data.iloc[idx]["transcript"]
        text = self.tokenizer(sentence).input_ids

        # run feature extractor
        audio_features = self.feature_extractor(
            audio_array, sampling_rate=self.sampling_rate, return_tensors="pt"
        )

        return audio_features, text


# Create a collator that will pad the audio features and text labels
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, batch):
        text_features = [{"input_ids": x[1]} for x in batch]
        batch_text = self.tokenizer.pad(
            text_features,
            return_tensors="pt",
        )
        audio_features = [{"input_features": x[0]["input_features"]} for x in batch]

        batch_audio = self.feature_extractor.pad(
            audio_features,
            return_tensors="pt",
        )
        batch_text["input_ids"] = batch_text["input_ids"].masked_fill(
            batch_text["attention_mask"].ne(1), -100
        )

        batch_audio["input_features"] = batch_audio["input_features"].squeeze(1)

        labels = batch_text["input_ids"].clone()
        if (labels[:, 0] == self.tokenizer.encode("")[0]).all().cpu().item():
            labels = labels[:, 1:]

        batch_text["labels"] = labels
        return batch_audio, batch_text


# Put into a lightning datamodule
class WhisperDataset(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0, whisper_model="tiny"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.whisper_model = whisper_model
        self.sampling_rate = 16_000

    def setup(self, stage=None):
        full_dataset = IMDA(self.data_dir, self.whisper_model)

        # Calculate sizes for train, val, and test sets
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = (
            torch.utils.data.random_split(
                full_dataset, [train_size, val_size, test_size]
            )
        )

        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            full_dataset.feature_extractor, full_dataset.tokenizer
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )


# Test if lightning datamodule working as intended
if __name__ == "__main__":
    dm = WhisperDataset(data_dir="data/")
    dm.setup()
    from tqdm import tqdm

    for batch in tqdm(dm.train_dataloader()):
        pass
