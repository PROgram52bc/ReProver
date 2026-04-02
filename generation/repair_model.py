"""Lightning module for the error repair model."""

import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional

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
        is_causal: bool = False,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.is_causal = is_causal

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if is_causal:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16
                if torch.cuda.is_available()
                else torch.float32,
                trust_remote_code=True,
            )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_causal:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss
        else:
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
