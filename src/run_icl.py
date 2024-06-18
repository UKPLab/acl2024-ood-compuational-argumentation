import os
import uuid

import click
import openai
import pandas
import wandb
from dotenv import load_dotenv
from nltk import word_tokenize
from rank_bm25 import BM25Okapi

from utils.composition import compose_instructions, LABEL_MAPPING, get_scores_for_row
from utils.prompting import update_plms, get_gpt_prompt_preds
from utils.training import check_run_done, get_metrics


@click.command()
@click.option('--task', type=str, default="x-stance")
@click.option('--model_name', type=str, default="gpt-3.5-turbo")
@click.option('--fold', type=int, default=0)
@click.option('--setup', type=str, default="cl")
@click.option('--k', type=int, default=4)
@click.option('--seed', type=int, default=0)
@click.option('--bm25_retrieval', type=bool, default=False)
def main(task, model_name, fold, setup, k, seed, bm25_retrieval):

    load_dotenv()

    task_id = task + "-" + setup + "-fold-" + str(fold)

    mode = os.getenv('MODE')
    training = "INSTRUCTION"

    openai.api_key = os.getenv('OPENAI_KEY')

    update_plms()

    hyperparameter = {
        "mode": mode,
        "model_name": model_name,
        "fold": fold,
        "setup": setup,
        "training": training,
        "k": k,
        "seed": seed,
    }

    template_indices = [0]

    is_run_done = check_run_done(task, hyperparameter)

    if not is_run_done:

        train_samples = pandas.read_json("../tasks/" + task_id + "/train.jsonl")
        test_samples = pandas.read_json("../tasks/" + task_id + "/test.jsonl")
        dev_samples = pandas.read_json("../tasks/" + task_id + "/dev.jsonl")

        if "text" in test_samples.columns:
            test_samples = test_samples.sort_values('text',key=lambda x:x.str.len())



        for i in template_indices:

            if bm25_retrieval:
                if "text" in train_samples.columns:
                    index_corpus = [word_tokenize(row["text"]) for i, row in train_samples.iterrows()]
                elif "hypothesis" in train_samples.columns:
                    index_corpus = [word_tokenize(row["hypothesis"] + " " + row["premise"]) for i, row in train_samples.iterrows()]
                else:
                    index_corpus = [word_tokenize(row["text_1"] + " " + row["text_2"]) for i, row in train_samples.iterrows()]

                bm25 = BM25Okapi(index_corpus)

                test_scores = [get_scores_for_row(bm25, row) for i, row in test_samples.iterrows()]
            else:
                test_scores = None

            token_label_mapping = dict([
                (token, label)
                for label, token in LABEL_MAPPING[task].items()
            ])

            dev_instructions = compose_instructions(train_samples, dev_samples, task, seed, k, i)
            test_instructions = compose_instructions(train_samples, test_samples, task, seed, k, i, scores=test_scores)


            dev_prediction, dev_prediction_token = get_gpt_prompt_preds(dev_instructions, token_label_mapping, model_name=model_name)
            test_prediction, test_prediction_token = get_gpt_prompt_preds(test_instructions, token_label_mapping, model_name=model_name)

            test_samples["pred"] = test_prediction
            test_samples["pred_token"] = test_prediction_token

            test_samples_table = wandb.Table(dataframe=test_samples)

            run_id = str(uuid.uuid4())

            wandb.init(
                entity="username",
                project="new-" + task,
                id=run_id,
                config={
                    k: str(v)
                    for k, v in hyperparameter.items()
                },
                tags=[training, mode]
            )

            wandb.log({
                "test_predictions": test_samples_table
            })

            test_f1, test_acc = get_metrics(test_samples["label"], test_prediction)
            dev_f1, dev_acc = get_metrics(dev_samples["label"], dev_prediction)

            metrics = {
                "eval/f1-macro": dev_f1,
                "test/f1-macro": test_f1,
                "eval/accuracy": dev_acc,
                "test/accuracy": test_acc,
            }

            wandb.log(metrics)

            wandb.config["template"] = str(i)
            wandb.config.update(
                {
                    k: str(v)
                    for k, v in hyperparameter.items()
                },
                allow_val_change=True
            )

            wandb.config["status"] = "done"
            wandb.join()

    else:
        print("Run already done")



if __name__ == "__main__":
    main()