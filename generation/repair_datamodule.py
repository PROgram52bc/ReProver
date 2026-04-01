"""Data module for the error repair model."""

import os
import json
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from typing import Dict, Any, List


class RepairDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: AutoTokenizer, max_inp_len: int, max_oup_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_inp_len = max_inp_len
        self.max_oup_len = max_oup_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.data[idx]
        
        input_encoding = self.tokenizer(
            entry["input"],
            max_length=self.max_inp_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        target_encoding = self.tokenizer(
            entry["target"],
            max_length=self.max_oup_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask padding tokens for loss calculation

        return {
            "input_ids": input_encoding["input_ids"].flatten(),
            "attention_mask": input_encoding["attention_mask"].flatten(),
            "labels": labels.flatten(),
        }


class ErrorRepairDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Path,
        tokenizer_name: str,
        batch_size: int,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        train_split_ratio: float = 0.9,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.train_split_ratio = train_split_ratio
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None) -> None:
        with open(self.dataset_path, "r") as f:
            data = json.load(f)

        # Split dataset into train and validation
        train_size = int(len(data) * self.train_split_ratio)
        train_data = data[:train_size]
        val_data = data[train_size:]

        self.train_dataset = RepairDataset(train_data, self.tokenizer, self.max_inp_seq_len, self.max_oup_seq_len)
        self.val_dataset = RepairDataset(val_data, self.tokenizer, self.max_inp_seq_len, self.max_oup_seq_len)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
        )
