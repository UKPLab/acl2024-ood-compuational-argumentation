import os
import uuid

import click
import torch
import wandb
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from models.classification import FrozenClassification
from models.encoding import Encoding
from utils.dataset import SimpleDataset
from utils.training import check_run_done


@click.command()
@click.option('--task', type=str, default="arg-cls")
@click.option('--model_name', type=str, default="bert-base-uncased")
@click.option('--fold', type=int, default=0)
@click.option('--setup', type=str, default="it")
@click.option('--pooling', type=str, default="mean")
@click.option('--seed', type=int, default=0)
@click.option('--batch_size', type=int, default=-1)
@click.option('--learning_rate', type=float, default=0.0005)
def main(task, model_name, fold, setup, pooling, seed, batch_size, learning_rate):

    load_dotenv()

    task_id = task + "-" + setup + "-fold-" + str(fold)

    mode = os.getenv('MODE')
    gpu = int(os.getenv('USE_CUDA'))

    training = "FROZEN"

    epochs = 20

    hyperparameter = {
        "mode": mode,
        "model_name": model_name,
        "pooling": pooling,
        "fold": fold,
        "setup": setup,
        "training": training,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
    }


    is_run_done = check_run_done(task, hyperparameter)

    if not is_run_done:

        text_encoding = Encoding(model_name=model_name, pooling_mode=pooling)

        train_dataset = SimpleDataset("../tasks/" + task_id + "/train.jsonl", task=task, text_encoding=text_encoding)
        dev_dataset = SimpleDataset("../tasks/" + task_id + "/dev.jsonl", task=task, text_encoding=text_encoding)
        test_dataset = SimpleDataset("../tasks/" + task_id + "/test.jsonl", task=task, text_encoding=text_encoding)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        num_labels = len(set(train_dataset.y))

        model = FrozenClassification(text_encoding.transforming.auto_model.config.hidden_size, num_labels, 0.1, hyperparameter)

        hyperparameter["warmup_steps"] =  int(len(train_dataloader) * epochs * 0.1)
        hyperparameter["training_steps"] =  int(len(train_dataloader) * epochs)

        run_id = str(uuid.uuid4())
        wandb_logger = WandbLogger(project="new-" + task, id=run_id)


        trainer = Trainer(
            max_epochs=20, gradient_clip_val=1.0, logger=wandb_logger, gpus=gpu,
            callbacks=[RichProgressBar(), ModelCheckpoint(monitor="eval/f1-macro",  mode="max", dirpath="./" + run_id + "-checkpoints")]

        )

        trainer.fit(model=model, train_dataloaders=[train_dataloader], val_dataloaders=[dev_dataloader])
        trainer.test(ckpt_path="best", dataloaders=[test_dataloader])
        test_predictions = trainer.predict(ckpt_path="best", dataloaders=[test_dataloader], return_predictions=True)
        test_predictions = torch.concat(test_predictions).numpy()

        test_samples = test_dataset.samples
        test_samples["pred"] = test_predictions

        test_samples_table = wandb.Table(dataframe=test_samples)

        dump_path = "dump/" + run_id
        os.system("mkdir -p " + dump_path)

        os.system("mv " + trainer.ckpt_path + " " + dump_path + "/model_dump.ckpt")

        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(dump_path + "/model_dump.ckpt")

        wandb.log_artifact(artifact)
        wandb.log({
            "test_predictions": test_samples_table
        })


        wandb.config["status"] = "done"

        wandb.config.update({
            k: str(v)
            for k, v in hyperparameter.items()
        })

        os.system("rm -rf ./" + run_id + "-checkpoints")
        os.system("rm -rf " + dump_path)
    else:
        print("Run already done")



if __name__ == "__main__":
    main()