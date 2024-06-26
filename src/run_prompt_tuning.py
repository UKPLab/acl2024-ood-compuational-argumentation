import os
import uuid

import click
import numpy
import pandas
import torch
import wandb
from dotenv import load_dotenv
from openprompt import PromptForClassification, PromptDataLoader
from openprompt.data_utils import InputFeatures
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.classification import SanityPromptClassification
from utils.dataset import convert_to_prompt_examples
from utils.prompting import STATIC_VERBALIZING, get_prompt_preds, get_model_family, TEMPLATES, CPUAutomaticVerbalizer, \
    new_collate_fct, update_plms
from utils.seed_util import seed_all
from utils.training import check_run_done, truncate_sentence


@click.command()
@click.option('--task', type=str, default="review")
@click.option('--model_name', type=str, default="microsoft/deberta-v3-base")
@click.option('--fold', type=int, default=0)
@click.option('--setup', type=str, default="ct")
@click.option('--seed', type=int, default=0)
@click.option('--batch_size', type=int, default=-1)
@click.option('--learning_rate', type=float, default=0.00002)
@click.option('--epochs', type=int, default=5)
@click.option('--verbalizing_mode', type=str, default="automatic")
def main(task, model_name, fold, setup, seed, batch_size, learning_rate, epochs, verbalizing_mode):

    load_dotenv()

    task_id = task + "-" + setup + "-fold-" + str(fold)

    mode = os.getenv('MODE')
    use_cuda = bool(int(os.getenv('USE_CUDA')))
    gpu = int(os.getenv('USE_CUDA'))

    training = "PROMPT_TUNING_PRE_SEARCH"

    model_family = get_model_family(model_name)

    update_plms()


    hyperparameter = {
        "mode": mode,
        "model_name": model_name,
        "fold": fold,
        "setup": setup,
        "training": training,
        "seed": seed,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        #"template_indices": template_indices,
        "verbalizing_mode": verbalizing_mode,
    }

    template_indices = [0]

    is_run_done = check_run_done(task, hyperparameter)

    if not is_run_done:

        train_samples = pandas.read_json("../tasks/" + task_id + "/train.jsonl")
        dev_samples = pandas.read_json("../tasks/" + task_id + "/dev.jsonl")
        test_samples = pandas.read_json("../tasks/" + task_id + "/test.jsonl")

        if "text" in dev_samples.columns:
            dev_samples = dev_samples.sort_values('text',key=lambda x:x.str.len())
            test_samples = test_samples.sort_values('text',key=lambda x:x.str.len())

        hyperparameter["warmup_steps"] = int(train_samples.shape[0] * epochs / batch_size * 0.1)
        hyperparameter["training_steps"] = int(train_samples.shape[0] * epochs / batch_size)

        num_classes = len(train_samples["label"].unique())

        test_predictions = []

        template_tokens = []

        for i in template_indices:

            seed_all(hyperparameter["seed"])

            plm, tokenizer, model_config, WrapperClass = load_plm(model_family, model_name)

            if "review" in task or "sentiment" in task:
                train_samples["text"] = train_samples["text"].apply(lambda sentence: truncate_sentence(sentence, 300, tokenizer))
                dev_samples["text"] = dev_samples["text"].apply(lambda sentence: truncate_sentence(sentence, 300, tokenizer))
                test_samples["text"] = test_samples["text"].apply(lambda sentence: truncate_sentence(sentence, 300, tokenizer))

            train_examples = convert_to_prompt_examples(train_samples)
            dev_examples = convert_to_prompt_examples(dev_samples)
            test_examples = convert_to_prompt_examples(test_samples)

            template_text = TEMPLATES[task][i]

            if not hasattr(tokenizer, 'sep_token') or tokenizer.sep_token is None:
                template_text = template_text.replace('{"special": "<sep>"}', '')

            prompt_template = ManualTemplate(
                text = template_text,
                tokenizer = tokenizer,
            )

            if verbalizing_mode == "static":
                prompt_verbalizing = ManualVerbalizer(
                    classes = list(range(num_classes)),
                    label_words = STATIC_VERBALIZING[task],
                    tokenizer = tokenizer,
                )
            elif verbalizing_mode == "automatic":
                prompt_verbalizing = CPUAutomaticVerbalizer(
                    tokenizer = tokenizer,
                    num_candidates=1000,
                    num_classes=num_classes,
                    label_word_num_per_class=50,
                )

            prompt_model = PromptForClassification(
                template = prompt_template,
                plm = plm,
                verbalizer = prompt_verbalizing
            )

            InputFeatures.collate_fct = new_collate_fct

            if use_cuda:
                prompt_model = prompt_model.cuda()

            if "bart" in model_name.lower() or "t5" in model_name.lower() or "t0" in model_name.lower():
                decoder_max_length = 3
            else:
                decoder_max_length = -1

            train_data_loader = PromptDataLoader(
                dataset = train_examples,
                tokenizer = tokenizer,
                template = prompt_template,
                tokenizer_wrapper_class=WrapperClass,
                batch_size=batch_size,
                shuffle=True,
                decoder_max_length=decoder_max_length
            )
            dev_data_loader = PromptDataLoader(
                dataset = dev_examples,
                tokenizer = tokenizer,
                template = prompt_template,
                tokenizer_wrapper_class=WrapperClass,
                batch_size=batch_size,
                shuffle=False,
                decoder_max_length=decoder_max_length
            )
            test_data_loader = PromptDataLoader(
                dataset = test_examples,
                tokenizer = tokenizer,
                template = prompt_template,
                tokenizer_wrapper_class=WrapperClass,
                batch_size=batch_size,
                shuffle=False,
                decoder_max_length=decoder_max_length
            )

            prompt_tuning_model = SanityPromptClassification(prompt_model=prompt_model, hyperparameter=hyperparameter)

            run_id = str(uuid.uuid4())

            wandb_logger = WandbLogger(project="new-" + task, id=run_id)

            trainer = Trainer(
                max_epochs=5, gradient_clip_val=1.0, logger=wandb_logger, gpus=gpu, num_sanity_val_steps=0,
                callbacks=[RichProgressBar(), ModelCheckpoint(monitor="eval/f1-macro",  mode="max", dirpath="./" + run_id + "-checkpoints")]
            )

            prompt_model.eval()

            train_prediction = get_prompt_preds(train_data_loader, prompt_model, use_cuda=use_cuda)

            if verbalizing_mode == "automatic":
                prompt_verbalizing.optimize_to_initialize()

            prompt_model.train()

            try:
                trainer.fit(model=prompt_tuning_model, train_dataloaders=[train_data_loader], val_dataloaders=[dev_data_loader])
                trainer.test(ckpt_path="best", dataloaders=[test_data_loader])
                test_prediction = trainer.predict(ckpt_path="best", dataloaders=[test_data_loader], return_predictions=True)

            except Exception as e:
                print(e)
                wandb.join()
                os.system("rm -rf ./" + run_id + "-checkpoints")
                return


            test_prediction = torch.concat(test_prediction).numpy()

            test_predictions.append(test_prediction)

            test_samples["pred"] = test_prediction

            test_samples_table = wandb.Table(dataframe=test_samples)

            wandb.log({
                "test_predictions": test_samples_table
            })

            if verbalizing_mode == "automatic":

                tokens = [prompt_verbalizing.tokenizer.convert_ids_to_tokens(i) for i in prompt_verbalizing.label_words_ids]

                tokens_table = wandb.Table(dataframe=pandas.DataFrame(numpy.array(tokens).T, columns=["label-" + str(i) for i in range(len(tokens))]))

                wandb.log({
                    "z_template-tokens": tokens_table
                })

    
            wandb.config["template"] = str(i)
            wandb.config.update(
                {
                    k: str(v)
                    for k, v in hyperparameter.items()
                },
                allow_val_change=True
            )

            wandb.config["status"] = "done"

            os.system("rm -rf ./" + run_id + "-checkpoints")

            wandb.join()

    else:
        print("Run already done")



if __name__ == "__main__":
    main()