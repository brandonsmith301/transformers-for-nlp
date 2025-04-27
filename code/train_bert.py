import numpy as np
import os
import time
import torch
import pandas as pd
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, BertConfig
import logging
import json
from datetime import datetime
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs
from copy import deepcopy
from tqdm.auto import tqdm

import hydra
from omegaconf import OmegaConf

import transformers
import datasets
from torch.utils.data import DataLoader

# --- custom model
from bert.transformer import *
from bert.loader import *
from bert.dataset import *

from utils.train_utils import *

logger = get_logger(__name__)


# --- for evaluation
def run_evaluation(accelerator, model, valid_dl, valid_ids):
    model.eval()

    all_predictions = []
    all_truths = []

    progress_bar = tqdm(
        range(len(valid_dl)), disable=not accelerator.is_local_main_process
    )

    for step, batch in enumerate(valid_dl):
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)
        predictions, references = accelerator.gather_for_metrics(
            (predictions, batch["labels"].to(torch.long).reshape(-1))
        )
        predictions, references = (
            predictions.cpu().numpy().tolist(),
            references.cpu().numpy().tolist(),
        )

        all_predictions.extend(predictions)
        all_truths.extend(references)

        progress_bar.update(1)
    progress_bar.close()

    # compute metric
    eval_dict = compute_metrics(all_predictions, all_truths)

    result_df = pd.DataFrame()
    result_df["id"] = valid_ids
    result_df["predictions"] = all_predictions
    result_df["truths"] = all_truths

    oof_df = deepcopy(result_df)
    oof_df = oof_df[["id", "predictions"]].copy()

    to_return = {
        "scores": eval_dict,
        "result_df": result_df,
        "oof_df": oof_df,
    }

    return to_return


@hydra.main(
    version_base=None,
    config_path="../conf/bert/",
    config_name="config_bert",
)
def run_training(config):
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision="bf16",
        device_placement=True,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        ],
    )
    config_dict = OmegaConf.to_container(config, resolve=True)

    # make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # print line func
    print_line = lambda: accelerator.print("#" + "~~" * 50 + "#")

    """
    set logging level to prevent duplicate logs from distributed training
    """
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # ------- Runtime Configs -----------------------------------------------------------#
    print_line()
    accelerator.print(f"setting seed: {config.seed}")
    set_seed(config.seed)

    if accelerator.is_main_process:
        os.makedirs(config.outputs.model_dir, exist_ok=True)
    print_line()

    # ------- load data -----------------------------------------------------------------#
    print_line()
    # Load IMDB dataset
    dataset = datasets.load_dataset("imdb")

    mlm = dataset["unsupervised"].to_pandas()
    dftr = dataset["train"].to_pandas()
    dfte = dataset["test"].to_pandas()

    dftr["id"] = dftr.index
    dfte["id"] = dfte.index

    valid_ids = dfte["id"]

    accelerator.print(f"shape of train data: {dftr.shape}")
    accelerator.print(f"{dftr.head()}")
    accelerator.print(f"shape of validation data: {dfte.shape}")

    with accelerator.main_process_first():
        dataset_creator = IMDBDataset(config)

        train_ds = dataset_creator.get_dataset(dftr)
        valid_ds = dataset_creator.get_dataset(dfte)

    tokenizer = dataset_creator.tokenizer

    data_collator = IMDBCollator(tokenizer=tokenizer, pad_to_multiple_of=64)
    data_collator_train = IMDBCollatorTrain(
        tokenizer=tokenizer, pad_to_multiple_of=64, kwargs=dict(cfg=config)
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=data_collator_train,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    accelerator.print("data preparation done...")
    print_line()

    # --- show batch -------------------------------------------------------------------#
    print_line()

    for b in train_dl:
        break
    show_batch(b, tokenizer, task="training", print_fn=accelerator.print)

    print_line()

    for b in valid_dl:
        break
    show_batch(b, tokenizer, task="training", print_fn=accelerator.print)

    print_line()

    # --- model -----------------------------------------------------------------------#
    # bert without pre-trained weights for fair comparison
    bert_config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        num_labels=2,
    )

    model = BertForSequenceClassification(bert_config)
    model.custom_config = config

    accelerator.print(f"Model loaded successfully")

    accelerator.wait_for_everyone()
    optimizer = model.configure_optim(device=accelerator.device)

    print_line()

    # --- Prepare -------------------------------------------------------------------#
    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = config.training.num_epochs
    grad_accumulation_steps = config.training.gradient_accumulation_steps
    warmup_pct = config.training.warmup_pct

    num_update_steps_per_epoch = len(train_dl) // grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct * num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print_line()

    # >--------------------------------------------------|
    # >-- training --------------------------------------|
    # >--------------------------------------------------|
    best_acc = -np.inf
    save_trigger = config.training.save_trigger

    patience_tracker = 0
    current_iteration = 0

    # Initialize log tracking
    training_logs = {
        "model_type": "bert",
        "config": OmegaConf.to_container(config, resolve=True),
        "training_history": [],
        "best_score": None,
        "best_epoch": None,
        "best_step": None,
        "training_time": None,
    }

    start_time = time.time()
    accelerator.wait_for_everyone()

    for epoch in range(num_epochs):
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(
            range(num_update_steps_per_epoch),
            disable=not accelerator.is_local_main_process,
        )
        loss_meter = AverageMeter()

        model.train()
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.training.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loss_meter.update(loss.item())

            if accelerator.sync_gradients:
                progress_bar.set_description(
                    f"STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                current_iteration += 1

            # >--------------------------------------------------|
            # >-- evaluation ------------------------------------|
            # >--------------------------------------------------|

            if (accelerator.sync_gradients) & (
                current_iteration % config.training.eval_frequency == 0
            ):
                # set model in eval mode
                model.eval()
                eval_response = run_evaluation(accelerator, model, valid_dl, valid_ids)

                scores_dict = eval_response["scores"]
                result_df = eval_response["result_df"]
                oof_df = eval_response["oof_df"]
                accuracy = scores_dict["accuracy"]

                print_line()
                et = as_minutes(time.time() - start_time)
                accelerator.print(
                    f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}"
                )
                print_line()
                accelerator.print(f">>> Current Best (ACC) = {round(accuracy, 4)}")

                print_line()

                # Log the evaluation results
                log_entry = {
                    "epoch": epoch + 1,
                    "step": step,
                    "total_step": current_iteration,
                    "time_elapsed": et,
                    "accuracy": accuracy,
                    "scores": scores_dict,
                    "learning_rate": get_lr(optimizer),
                    "loss": loss_meter.avg,
                }
                training_logs["training_history"].append(log_entry)

                is_best = False
                if accuracy >= best_acc:
                    best_acc = accuracy
                    is_best = True
                    patience_tracker = 0

                    best_dict = dict()
                    for k, v in scores_dict.items():
                        best_dict[f"{k}_at_best"] = v

                    # Update best scores in logs
                    training_logs["best_score"] = accuracy
                    training_logs["best_epoch"] = epoch + 1
                    training_logs["best_step"] = current_iteration
                    training_logs["best_metrics"] = best_dict
                else:
                    patience_tracker += 1

                if is_best:  # do in main process
                    oof_df.to_csv(
                        os.path.join(config.outputs.model_dir, f"oof_df_best.csv"),
                        index=False,
                    )
                    result_df.to_csv(
                        os.path.join(config.outputs.model_dir, f"result_df_best.csv"),
                        index=False,
                    )

                    # save log when we have a new best model
                    if accelerator.is_main_process:
                        training_logs["training_time"] = et
                        log_file = save_training_log(
                            config, training_logs, is_best=True
                        )
                        accelerator.print(f">>> Saved best model log to {log_file}")
                else:
                    accelerator.print(
                        f">>> patience reached {patience_tracker}/{config.training.patience}"
                    )
                    accelerator.print(f">>> current best score: {round(best_acc, 4)}")

                oof_df.to_csv(
                    os.path.join(config.outputs.model_dir, f"oof_df_last.csv"),
                    index=False,
                )
                result_df.to_csv(
                    os.path.join(config.outputs.model_dir, f"result_df_last.csv"),
                    index=False,
                )

                # Save the current training log regardless of best or not
                if accelerator.is_main_process:
                    training_logs["training_time"] = et
                    save_training_log(config, training_logs, is_best=False)

                # saving -----
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                unwrapped_model.save_pretrained(
                    f"{config.outputs.model_dir}/last",
                    state_dict=accelerator.get_state_dict(model),
                    save_function=accelerator.save,
                )

                if accelerator.is_main_process:
                    tokenizer.save_pretrained(f"{config.outputs.model_dir}/last")

                if best_acc > save_trigger:
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(f"{config.outputs.model_dir}/best")
                    unwrapped_model.save_pretrained(
                        f"{config.outputs.model_dir}/best",
                        state_dict=accelerator.get_state_dict(model),
                        save_function=accelerator.save,
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(f"{config.outputs.model_dir}/best")

                # -- post eval
                model.train()
                torch.cuda.empty_cache()
                print_line()

                # early stopping ----
                if patience_tracker >= config.training.patience:
                    print("stopping early")
                    model.eval()
                    accelerator.end_training()
                    return

    # Save final training log at the end
    if accelerator.is_main_process:
        training_logs["training_time"] = as_minutes(time.time() - start_time)
        save_training_log(config, training_logs, is_best=False)

    # --- end training
    accelerator.end_training()


if __name__ == "__main__":
    run_training()
