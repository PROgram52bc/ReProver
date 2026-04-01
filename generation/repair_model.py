"""Lightning module for the error repair model."""

import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AutoTokenizer
from typing import Dict, Any

from common import get_optimizers

torch.set_float32_matmul_precision("medium")


class ErrorRepairModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch["input_ids"]),
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        loss = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        self.log(
            "loss_val",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch["input_ids"]),
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )
