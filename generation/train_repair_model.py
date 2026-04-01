import os
import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

from generation.repair_model import ErrorRepairModel
from generation.repair_datamodule import ErrorRepairDataModule

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a T5-based model for error repair."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the curated dataset of failed tactics and ground truths.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/byt5-small",
        help="Hugging Face model name for the T5 architecture.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-inp-seq-len", type=int, default=1024)
    parser.add_argument("--max-oup-seq-len", type=int, default=256)
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/repair_model_train",
        help="Directory to save logs and checkpoints.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="error_repair_experiment",
        help="Experiment name for logging.",
    )

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # Setup data module
    data_module = ErrorRepairDataModule(
        dataset_path=Path(args.dataset_path),
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_inp_seq_len=args.max_inp_seq_len,
        max_oup_seq_len=args.max_oup_seq_len,
    )

    # Setup model
    model = ErrorRepairModel(
        model_name=args.model_name,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_inp_seq_len=args.max_inp_seq_len,
        max_oup_seq_len=args.max_oup_seq_len,
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        filename="repair-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="loss_val",
        mode="min",
    )

    # Logger
    wandb_logger = WandbLogger(project="reprover_error_repair", name=args.exp_name, save_dir=args.log_dir)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.num_gpus > 0 else "cpu",
        devices=args.num_gpus if args.num_gpus > 0 else "auto",
        callbacks=[lr_monitor, checkpoint_callback],
        logger=wandb_logger,
        # precision="16-mixed", # Uncomment for mixed precision training if supported
    )

    logger.info("Starting training for the error repair model...")
    trainer.fit(model, datamodule=data_module)
    logger.info("Training finished.")

    # Save the final model and tokenizer
    final_model_path = Path(args.log_dir) / "final_repair_model"
    model.model.save_pretrained(final_model_path)
    model.tokenizer.save_pretrained(final_model_path)
    logger.info(f"Final model and tokenizer saved to {final_model_path}")

if __name__ == "__main__":
    main()
