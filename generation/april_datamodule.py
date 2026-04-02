"""Data module for the APRIL proof repair dataset."""

import os
import json
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from typing import Dict, Any, List, Optional
from loguru import logger


class AprilDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_inp_len: int,
        max_oup_len: int,
        is_causal: bool = False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_inp_len = max_inp_len
        self.max_oup_len = max_oup_len
        self.is_causal = is_causal

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.data[idx]

        if self.is_causal:
            # Format for Causal LM (similar to gAPRIL chat template)
            system_prompt = (
                "You are a Lean 4 programmer diagnosing a single failing proof. "
                "Assume you only see the incorrect proof text, the infoview state"
                " near the failure, and Lean's error message."
            )
            user_prompt = f"**Instruction:** Provide the full corrected Lean 4 theorem/proof in a single ```lean``` code block.\n\n**Context:**\n\nIncorrect proof:\n```lean\n{entry['incorrect_proof']}\n```\n\nInfoview state:\n{entry['state_at_error']}\n\nLine at error:\n{entry['line_at_error']}\n\nLean error:\n{entry['error']}"
            assistant_response = f"Explanation: {entry['explanation']}\nFix: {entry['fix_suggestion']}\nCorrected Proof:\n```lean\n{entry['correct_proof']}\n```"

            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ]
            
            # For training causal models, we usually concatenate everything
            full_text = self.tokenizer.apply_chat_template(chat, tokenize=False)
            
            encodings = self.tokenizer(
                full_text,
                max_length=self.max_inp_len + self.max_oup_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            labels = encodings["input_ids"].clone()
            # We should ideally mask the prompts in labels, but for simplicity:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": encodings["input_ids"].flatten(),
                "attention_mask": encodings["attention_mask"].flatten(),
                "labels": labels.flatten(),
            }

        else:
            # Format for T5
            input_text = f"State: {entry['state_at_error']} | Bad Tactic: {entry['line_at_error']} | Error: {entry['error']}"
            target_text = entry["correct_proof"]  # Or extract specific tactic if possible

            input_encoding = self.tokenizer(
                input_text,
                max_length=self.max_inp_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            target_encoding = self.tokenizer(
                target_text,
                max_length=self.max_oup_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            labels = target_encoding["input_ids"]
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": input_encoding["input_ids"].flatten(),
                "attention_mask": input_encoding["attention_mask"].flatten(),
                "labels": labels.flatten(),
            }


class AprilRepairDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        tokenizer_name: str,
        batch_size: int,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.is_causal = is_causal
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None) -> None:
        def load_data(data_dir: Path, split: str) -> List[Dict[str, Any]]:
            all_data = []
            # Find all jsonl files recursively that match the split name
            jsonl_files = list(data_dir.rglob("*.jsonl"))
            
            # Target split names to match
            target_splits = [split]
            if split == "val":
                target_splits.append("validation")
            
            for path in jsonl_files:
                if any(s in path.name.lower() for s in target_splits):
                    logger.info(f"Loading {split} split data from {path}...")
                    with open(path, "r") as f:
                        all_data.extend([json.loads(line) for line in f])
            
            # Also check for .json files just in case
            json_files = list(data_dir.rglob("*.json"))
            for path in json_files:
                if any(s in path.name.lower() for s in target_splits):
                    logger.info(f"Loading {split} split data from {path}...")
                    with open(path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
            
            if not all_data:
                logger.warning(f"Could not find any {split} data in {data_dir}")
            else:
                logger.info(f"Total entries loaded for {split}: {len(all_data)}")
                
            return all_data

        train_data = load_data(self.data_dir, "train")
        val_data = load_data(self.data_dir, "val")

        if not train_data:
            logger.error(f"Train data is empty in {self.data_dir}")
            return

        self.train_dataset = AprilDataset(
            train_data, self.tokenizer, self.max_inp_seq_len, self.max_oup_seq_len, self.is_causal
        )
        self.val_dataset = AprilDataset(
            val_data, self.tokenizer, self.max_inp_seq_len, self.max_oup_seq_len, self.is_causal
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() or 1,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 1,
        )
