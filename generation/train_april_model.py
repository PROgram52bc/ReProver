import os
import argparse
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

from generation.repair_model import ErrorRepairModel
from generation.april_datamodule import AprilRepairDataModule

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a proof repair model using the APRIL dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing the APRIL dataset (train.json, val.json).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/byt5-small",
        help="Hugging Face model name for the architecture.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-inp-seq-len", type=int, default=1024)
    parser.add_argument("--max-oup-seq-len", type=int, default=1024)
    parser.add_argument(
        "--is-causal",
        action="store_true",
        help="Whether the model is a causal language model (e.g., Llama, GPT).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/april_model_train",
        help="Directory to save logs and checkpoints.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="april_repair_experiment",
        help="Experiment name for logging.",
    )

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # Setup data module
    data_module = AprilRepairDataModule(
        data_dir=Path(args.data_dir),
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_inp_seq_len=args.max_inp_seq_len,
        max_oup_seq_len=args.max_oup_seq_len,
        is_causal=args.is_causal,
    )

    # Setup model
    model = ErrorRepairModel(
        model_name=args.model_name,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_inp_seq_len=args.max_inp_seq_len,
        max_oup_seq_len=args.max_oup_seq_len,
        is_causal=args.is_causal,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        filename="april-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="loss_val",
        mode="min",
    )

    # Logger
    wandb_logger = WandbLogger(project="reprover_april_repair", name=args.exp_name, save_dir=args.log_dir)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.num_gpus > 0 else "cpu",
        devices=args.num_gpus if args.num_gpus > 0 else "auto",
        callbacks=[lr_monitor, checkpoint_callback],
        logger=wandb_logger,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision="bf16-mixed" if torch.cuda.is_available() else "32",
    )

    logger.info("Starting training for the APRIL proof repair model...")
    trainer.fit(model, datamodule=data_module)
    logger.info("Training finished.")

    # Save the final model and tokenizer
    final_model_path = Path(args.log_dir) / "final_april_model"
    model.model.save_pretrained(final_model_path)
    model.tokenizer.save_pretrained(final_model_path)
    logger.info(f"Final model and tokenizer saved to {final_model_path}")

if __name__ == "__main__":
    main()
