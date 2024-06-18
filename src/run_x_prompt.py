import os
import uuid

import click
import numpy
import pandas
import wandb
from dotenv import load_dotenv
from openprompt import PromptForClassification, PromptDataLoader
from openprompt.data_utils import InputFeatures
from openprompt.plms import load_plm
from openprompt.prompts import ManualVerbalizer

from models.templates import XManualTemplate
from utils.dataset import convert_to_prompt_examples
from utils.prompting import get_prompt_preds, get_model_family, CPUAutomaticVerbalizer, \
    new_collate_fct, update_plms, X_TEMPLATES, X_STATIC_VERBALIZING
from utils.training import check_run_done, get_metrics, truncate_sentence


@click.command()
@click.option('--task', type=str, default="x-sentiment")
@click.option('--model_name', type=str, default="bert-base-multilingual-uncased")
@click.option('--fold', type=int, default=1)
@click.option('--setup', type=str, default="cl")
@click.option('--seed', type=int, default=0)
@click.option('--batch_size', type=int, default=-1)
@click.option('--verbalizing_mode', type=str, default="automatic")
def main(task, model_name, fold, setup, seed, batch_size, verbalizing_mode):

    load_dotenv()

    task_id = task + "-" + setup + "-fold-" + str(fold)

    pred_store = os.getenv('PRED_STORE')
    mode = os.getenv('MODE')
    use_cuda = bool(int(os.getenv('USE_CUDA')))

    training = "IN-CONTEXT_PROMPTING"

    model_family = get_model_family(model_name)

    update_plms()


    hyperparameter = {
        "mode": mode,
        "model_name": model_name,
        "fold": fold,
        "setup": setup,
        "training": training,
        "seed": seed,
        #"template_indices": template_indices,
        "verbalizing_mode": verbalizing_mode,
    }

    template_indices = [0]

    is_run_done = check_run_done(task, hyperparameter)

    if not is_run_done:

        plm, tokenizer, model_config, WrapperClass = load_plm(model_family, model_name)

        train_samples = pandas.read_json("../tasks/" + task_id + "/train.jsonl")
        dev_samples = pandas.read_json("../tasks/" + task_id + "/dev.jsonl")
        test_samples = pandas.read_json("../tasks/" + task_id + "/test.jsonl")

        if "review" in task or "sentiment" in task:
            train_samples["text"] = train_samples["text"].apply(lambda sentence: truncate_sentence(sentence, 300, tokenizer))
            dev_samples["text"] = dev_samples["text"].apply(lambda sentence: truncate_sentence(sentence, 300, tokenizer))
            test_samples["text"] = test_samples["text"].apply(lambda sentence: truncate_sentence(sentence, 300, tokenizer))

        num_classes = len(train_samples["label"].unique())

        train_examples = convert_to_prompt_examples(train_samples)
        dev_examples = convert_to_prompt_examples(dev_samples)
        test_examples = convert_to_prompt_examples(test_samples)

        for i in template_indices:

            template_texts = X_TEMPLATES[task]

            prompt_template = XManualTemplate(
                text = template_texts,
                tokenizer = tokenizer,
            )

            if verbalizing_mode == "static":
                prompt_verbalizing = ManualVerbalizer(
                    classes = list(range(num_classes)),
                    label_words = X_STATIC_VERBALIZING[task],
                    tokenizer = tokenizer,
                )
            elif verbalizing_mode == "automatic":
                prompt_verbalizing = CPUAutomaticVerbalizer(
                    tokenizer = tokenizer,
                    num_candidates=1000,
                    num_classes=num_classes,
                    label_word_num_per_class=50
                )


            prompt_model = PromptForClassification(
                template = prompt_template,
                plm = plm,
                verbalizer = prompt_verbalizing,
                plm_eval_mode = True
            )

            if use_cuda:
                prompt_model = prompt_model.cuda()

            if "bart" in model_name.lower() or "t5" in model_name.lower() or "t0" in model_name.lower():
                decoder_max_length = 3
            else:
                decoder_max_length = -1

            InputFeatures.collate_fct = new_collate_fct

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
            wandb.config["template"] = str(i)

            train_prediction = get_prompt_preds(train_data_loader, prompt_model, use_cuda=use_cuda)

            if verbalizing_mode == "automatic":
                prompt_verbalizing.optimize_to_initialize()

            dev_prediction = get_prompt_preds(dev_data_loader, prompt_model, use_cuda=use_cuda)
            test_prediction = get_prompt_preds(test_data_loader, prompt_model, use_cuda=use_cuda)


            test_samples["pred"] = test_prediction

            test_samples_table = wandb.Table(dataframe=test_samples)

            wandb.log({
                "test_predictions": test_samples_table
            })

            train_f1, train_acc = get_metrics(train_samples["label"], train_prediction)
            dev_f1, dev_acc = get_metrics(dev_samples["label"], dev_prediction)
            test_f1, test_acc = get_metrics(test_samples["label"], test_prediction)

            metrics = {
                "train/f1-macro": train_f1,
                "eval/f1-macro": dev_f1,
                "test/f1-macro": test_f1,
                "train/accuracy": train_acc,
                "eval/accuracy": dev_acc,
                "test/accuracy": test_acc,
            }

            wandb.log(metrics)

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
            wandb.join()



            os.system("rm -rf ./" + run_id + "-checkpoints")

    else:
        print("Run already done")



if __name__ == "__main__":
    main()