import wandb
from sklearn.metrics import f1_score, accuracy_score
from wandb.apis.public import Runs



def compute_metrics(eval_preds):
    logits, labels = eval_preds

    if type(logits) == tuple:
        ## for BART model
        logits = logits[0]

    predictions = numpy.argmax(logits, axis=-1)

    metrics = {
        "accuracy": accuracy_score(y_true=labels, y_pred=predictions),
        "f1-macro": f1_score(y_true=labels, y_pred=predictions, average="macro"),
    }

    return metrics

def truncate_sentence(sentence, truncation_length, tokenizer):
    if sentence == None:
        sentence = "None"
    tokens = tokenizer.encode(text=sentence, max_length=truncation_length, truncation=True, add_special_tokens=False)
    return tokenizer.decode(tokens)


def check_run_done(task, hyperparameter):
    api = wandb.Api()

    lookup_filter = {
        "config." + k: str(v)
        for k, v in hyperparameter.items()
    }
    lookup_filter["config.status"] = "done"
    #lookup_filter["config.mode"] = "prod"

    try:
        lookup_runs = list(Runs(entity="username", project="new-" + task, filters=lookup_filter, client=api.client))
    except:
        return False

    if len(lookup_runs) == 0:
        del lookup_filter["config.status"]
        try:
            lookup_runs = list(Runs(entity="username", project="new-" + task, filters=lookup_filter, client=api.client))
        except:
            return False

    return len(lookup_runs) > 0

def get_runs(task, hyperparameter, project_prefix):
    api = wandb.Api()

    lookup_filter = {
        "config." + k: str(v)
        for k, v in hyperparameter.items()
    }
    lookup_filter["config.status"] = "done"

    return list(Runs(entity="username", project=project_prefix + task, filters=lookup_filter, client=api.client))


def get_pre_init_run(task, hyperparameter, steps=0):
    api = wandb.Api()

    lookup_filter = {
        "config." + k: str(v)
        for k, v in hyperparameter.items()
        if k not in ["learning_rate", "batch_size", "dropout_rate", "warmup_steps", "training_steps", "mode"]
    }
    if steps == 0:
        lookup_filter["config.training"] = "FROZEN"
    else:
        lookup_filter["config.training"] = "FROZEN_SINGLE_STEP"
        lookup_filter["config.steps"] = str(steps)

    lookup_filter["config.status"] = "done"

    lookup_runs = list(Runs(entity="username", project="new-" + task, filters=lookup_filter, client=api.client))

    return lookup_runs

def get_metrics(labels, preds):
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)

    return f1, acc
