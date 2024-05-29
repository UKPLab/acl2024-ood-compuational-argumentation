import os
import traceback
import uuid

import click
import pandas
import torch
import wandb
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from models.classification import MeanFineTuningClassification, CLSFineTuningClassification
from utils.composition import compose_samples
from utils.dataset import FineTuningDataset
from utils.training import check_run_done, truncate_sentence


@click.command()
@click.option('--task', type=str, default="arg-q")
@click.option('--model_name', type=str, default="microsoft/deberta-v3-base")
@click.option('--fold', type=int, default=0)
@click.option('--setup', type=str, default="ct")
@click.option('--pooling', type=str, default="cls")
@click.option('--seed', type=int, default=0)
@click.option('--batch_size', type=int, default=-1)
@click.option('--epochs', type=int, default=5)
@click.option('--dropout_rate', type=float, default=0.1)
@click.option('--learning_rate', type=float, default=0.00002)
@click.option('--dump', type=bool, default=False)
def main(task, model_name, fold, setup, pooling, seed, batch_size, epochs, dropout_rate, learning_rate, dump):

    if batch_size == -1:
        batch_size = 16

    load_dotenv()

    task_id = task + "-" + setup + "-fold-" + str(fold)

    if "few-shot" in task:
        base_task = task.split("@")[-1]
    else:
        base_task = task

    if dump:
        task = "dump-" + task

    mode = os.getenv('MODE')
    gpu = int(os.getenv('USE_CUDA'))

    training = "FINE_TUNING"

    train_samples = pandas.read_json("../tasks/" + task_id + "/train.jsonl")
    dev_samples = pandas.read_json("../tasks/" + task_id + "/dev.jsonl")
    test_samples = pandas.read_json("../tasks/" + task_id + "/test.jsonl")

    if "text" in dev_samples.columns:
        dev_samples = dev_samples.sort_values('text',key=lambda x:x.str.len())
        test_samples = test_samples.sort_values('text',key=lambda x:x.str.len())


    num_classes = len(train_samples["label"].unique())

    hyperparameter = {
        "mode": mode,
        "model_name": model_name,
        "pooling": pooling,
        "fold": fold,
        "setup": setup,
        "training": training,
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "seed": seed,
    }

    hyperparameter["warmup_steps"] = int(train_samples.shape[0] * epochs / batch_size * 0.1)
    hyperparameter["training_steps"] = int(train_samples.shape[0] * epochs / batch_size)

    is_run_done = check_run_done(task, hyperparameter)

    if not is_run_done:

        if pooling == "mean":
            model = MeanFineTuningClassification(hyperparameter=hyperparameter, num_classes=num_classes)
        elif pooling == "cls":
            model = CLSFineTuningClassification(hyperparameter=hyperparameter, num_classes=num_classes)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if "review" in task or "sentiment" in task:
            train_samples["text"] = train_samples["text"].apply(lambda sentence: truncate_sentence(sentence, 300, tokenizer))
            dev_samples["text"] = dev_samples["text"].apply(lambda sentence: truncate_sentence(sentence, 300, tokenizer))
            test_samples["text"] = test_samples["text"].apply(lambda sentence: truncate_sentence(sentence, 300, tokenizer))

        def tokenize_function(samples):
            composed_samples = compose_samples(samples, task=base_task, sep_token=tokenizer.sep_token)
            return [
                tokenizer.encode(composed_sample, truncation=True)
                for composed_sample in composed_samples
            ]

        train_samples["input_ids"] = tokenize_function(train_samples)
        dev_samples["input_ids"] = tokenize_function(dev_samples)
        test_samples["input_ids"] = tokenize_function(test_samples)

        train_dataset = FineTuningDataset(train_samples)
        dev_dataset = FineTuningDataset(dev_samples)
        test_dataset = FineTuningDataset(test_samples)

        run_id = str(uuid.uuid4())

        wandb_logger = WandbLogger(project="new-" + task, id=run_id)




        if "xnli" in task or "x-sentiment":
            batch_size = int(batch_size/4)
            accumulate_grad_batches = 4
        else:
            accumulate_grad_batches = 1

        trainer = Trainer(
            max_epochs=epochs, gradient_clip_val=1.0, logger=wandb_logger, gpus=gpu, num_sanity_val_steps=0, accumulate_grad_batches=accumulate_grad_batches,
            callbacks=[RichProgressBar(), ModelCheckpoint(monitor="eval/f1-macro",  mode="max", dirpath="./" + run_id + "-checkpoints")]

        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=512, padding="longest")

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

        try:
            trainer.fit(model=model, train_dataloaders=[train_dataloader], val_dataloaders=[dev_dataloader])
            trainer.test(ckpt_path="best", dataloaders=[test_dataloader])
            test_prediction = trainer.predict(ckpt_path="best", dataloaders=[test_dataloader], return_predictions=True)

        except Exception as e:
            print(e)
            traceback.print_exc()
            wandb.join()
            os.system("rm -rf ./" + run_id + "-checkpoints")
            return

        test_prediction = torch.concat(test_prediction).numpy()

        test_samples["pred"] = test_prediction


        test_samples_table = wandb.Table(dataframe=test_samples)

        wandb.log({
            "test_predictions": test_samples_table
        })

        #os.system("mv " + trainer.state.best_model_checkpoint + "/* " + model_store + "/" + run_id)

        wandb.config["status"] = "done"
        wandb.config.update(
            {
                k: str(v)
                for k, v in hyperparameter.items()
            },
            allow_val_change=True
        )

        if dump:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(trainer.checkpoint_callback.best_model_path)
            wandb.log_artifact(artifact)

        os.system("rm -rf ./" + run_id + "-checkpoints")
    else:
        print("Run already done")



if __name__ == "__main__":
    main()