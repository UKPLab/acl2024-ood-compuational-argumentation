from nltk import word_tokenize
from openprompt.data_utils import InputExample
from tqdm import tqdm


def compose_samples(samples, task, sep_token=None):
    composed_samples = COMPOSITION[task](samples, sep_token=sep_token)
    return composed_samples

def compose_text_topic_samples(samples, sep_token):

    if sep_token is not None:
        composed_samples = samples.apply(
            lambda row: row["text"] + " " + sep_token + " " + row["topic"]
            , axis=1)
    else:
        composed_samples = samples.apply(
            lambda row: row["text"] + " " + row["topic"]
            , axis=1)

    return composed_samples

def compose_text_target_samples(samples, sep_token):

    if sep_token is not None:
        composed_samples = samples.apply(
            lambda row: row["text"] + " " + sep_token + " " + row["target"]
            , axis=1)
    else:
        composed_samples = samples.apply(
            lambda row: row["text"] + " " + row["target"]
            , axis=1)

    return composed_samples

def compose_premise_choice_samples(samples, sep_token):

    if sep_token is not None:
        composed_samples = samples.apply(
            lambda row: row["premise"] + " " + sep_token + " " + row["choice"]
            , axis=1)
    else:
        composed_samples = samples.apply(
            lambda row: row["premise"] + " " + row["choice"]
            , axis=1)

    return composed_samples


def compose_entailment_samples(samples, sep_token):

    if sep_token is not None:
        composed_samples = samples.apply(
            lambda row: row["text_1"] + " " + sep_token + " " + row["text_2"]
            , axis=1)
    else:
        composed_samples = samples.apply(
            lambda row: row["text_1"] + " " + row["text_2"]
            , axis=1)

    return composed_samples

def compose_premise_hypothesis_samples(samples, sep_token):

    if sep_token is not None:
        composed_samples = samples.apply(
            lambda row: row["premise"] + " " + sep_token + " " + row["hypothesis"]
            , axis=1)
    else:
        composed_samples = samples.apply(
            lambda row: row["premise"] + " " + row["hypothesis"]
            , axis=1)

    return composed_samples


def compose_ukp_a_samples(samples, sep_token):

    if sep_token is not None:
        composed_samples = samples.apply(
            lambda row: row["text_1"] + " " + row["text_2"] + " [SEP] " + row["topic"]
            , axis=1)
    else:
        composed_samples = samples.apply(
            lambda row: row["text_1"] + " " + row["text_2"] + " " + row["topic"]
            , axis=1)

    return composed_samples

def compose_pair_topic_samples(samples, sep_token):

    if sep_token is not None:
        composed_samples = samples.apply(
            lambda row: row["text_1"] + " " + row["text_2"] + " [SEP] " + row["topic"]
            , axis=1)
    else:
        composed_samples = samples.apply(
            lambda row: row["text_1"] + " " + row["text_2"] + " " + row["topic"]
            , axis=1)

    return composed_samples

def compose_ag_news_samples(samples, sep_token):
    composed_samples = samples.apply(
        lambda row: row["headline"] + " " + row["text"]
        , axis=1)
    return composed_samples

def compose_review_samples(samples, sep_token):
    composed_samples = samples.apply(
        lambda row: row["text"]
        , axis=1)
    return composed_samples


COMPOSITION = {
    "arg-cls": compose_text_topic_samples,
    "entail": compose_entailment_samples,
    "arg-qua": compose_pair_topic_samples,
    "arg-sim": compose_pair_topic_samples,
    "evi-cls": compose_text_topic_samples,
    "stance": compose_text_target_samples,
    "x-stance": compose_text_topic_samples,
    "x-sentiment": compose_review_samples,
    "review": compose_review_samples,
}



LABEL_MAPPING = {
    "arg-cls": {
        0: "neutral",
        1: "favor",
        2: "against",
    },
    "stance": {
        0: "favor",
        1: "against",
        2: "neutral",
    },
    "entail": {
        0: "yes",
        1: "no",
    },
    "review": {
        0: "negative",
        1: "positive",
    },
    "arg-qua": {
        0: "first",
        1: "second",
    },
    "arg-sim": {
        0: "no",
        1: "yes",
    },
    "evi-cls": {
        0: "no",
        1: "yes",
    },
}

X_LABEL_MAPPING = {
    "x-stance": {
        "de": {
            0: "pro",
            1: "kontro"
        },
        "fr": {
            0: "pour",
            1: "contre"
        },
        "it": {
            0: "favore",
            1: "contro"
        },
    },
    "x-sentiment": {
        "en": {
            0: "negative",
            1: "positive"
        },
        "de": {
            0: "negativ",
            1: "positiv"
        },
        "fr": {
            0: "négative",
            1: "positive"
        },
        "jp": {
            0: "ネガティブ",
            1: "ポジティブ"
        },
    }
}
def compose_stance_example_instruction_1(instruction_example_samples, example_label=True):
    instruction = "What is the attitude of the following text regarding the given topic."

    if example_label:
        instruction += " Options are neutral, favor, or against."

    instruction += "\n\n"

    for i, row in instruction_example_samples.iterrows():
        instruction += "Input: " + row["text"] + "\n"
        instruction += "Topic: " + row["topic"] + "\n"
        instruction += "Label: " + row["label"] + "\n\n"

    return instruction

def ukp_argmin_instruction(example_label):
    instruction = "What is the attitude of the following argument regarding the given topic?"

    if example_label:
        instruction += " Options are neutral, favor, or against."

    instruction += "\n"

    return instruction

def stance_instruction(example_label):
    instruction = "What is the attitude of the following text regarding the given target?"

    if example_label:
        instruction += " Options are neutral, favor, or against."

    instruction += "\n"

    return instruction


def de_x_stance_instruction(example_label):
    instruction = "Welche Einstellung hat der folgende Text zum gegebenen Thema?"

    if example_label:
        instruction += " Optionen sind pro oder kontra."

    instruction += "\n"

    return instruction

def fr_x_stance_instruction(example_label):
    instruction = "Quelle est l'attitude du texte suivant par rapport au sujet donné?"

    if example_label:
        instruction += " Les options sont pour ou contre."

    instruction += "\n"

    return instruction

def it_x_stance_instruction(example_label):
    instruction = "Qual è l'atteggiamento del seguente testo rispetto all'argomento dato?"

    if example_label:
        instruction += " Le opzioni sono a favore o contro."

    instruction += "\n"

    return instruction




def de_x_stance_instruction(example_label):
    instruction = "Welche Einstellung hat der folgende Text zum gegebenen Thema?"

    if example_label:
        instruction += " Optionen sind pro oder kontra."

    instruction += "\n"

    return instruction

def fr_x_stance_instruction(example_label):
    instruction = "Quelle est l'attitude du texte suivant par rapport au sujet donné?"

    if example_label:
        instruction += " Les options sont pour ou contre."

    instruction += "\n"

    return instruction

def it_x_stance_instruction(example_label):
    instruction = "Qual è l'atteggiamento del seguente testo rispetto all'argomento dato?"

    if example_label:
        instruction += " Le opzioni sono a favore o contro."

    instruction += "\n"

    return instruction



def review_instruction(example_label):
    instruction = "What is the sentiment of the following text?"

    if example_label:
        instruction += " Options are positive or negative."

    instruction += "\n"

    return instruction



def entailment_instruction(example_label):
    instruction = "Can we conclude an entailment from the following two texts?"

    if example_label:
        instruction += " Options are yes or no."

    instruction += "\n"

    return instruction

def evi_sen_instruction(example_label):
    instruction = "Corresponds the following evidence to the given topic?"

    if example_label:
        instruction += " Options are yes or no."

    instruction += "\n"

    return instruction


def ukp_a_instruction(example_label):
    instruction = "Are the following arguments similar regarding the given topic?"

    if example_label:
        instruction += " Options are yes or no."

    instruction += "\n"

    return instruction



def arg_q_instruction(example_label):
    instruction = "Given the following two arguments and the topic they cover, which one has the higher quality?"

    if example_label:
        instruction += " Options are first or second."

    instruction += "\n"

    return instruction


def compose_ukp_argmin_topic_demonstration_1(task, demonstration_example_samples):
    demonstration = ""

    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += INSTRUCTION[task](example_label=True)
        demonstration += "Argument: " + row["text"] + "\n"
        demonstration += "Topic: " + row["topic"] + "\n"
        demonstration += "Label: " + label + "\n\n"

    return demonstration


def compose_review_demonstration_1(task, demonstration_example_samples):
    demonstration = ""

    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += INSTRUCTION[task](example_label=True)
        demonstration += "Review: " + row["text"] + "\n"
        demonstration += "Label: " + label + "\n\n"

    return demonstration

def compose_entailment_demonstration_1(task, demonstration_example_samples):
    demonstration = ""

    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += INSTRUCTION[task](example_label=True)
        demonstration += "Text 1: " + row["text_1"] + "\n"
        demonstration += "Text 2: " + row["text_2"] + "\n"
        demonstration += "Label: " + label + "\n\n"

    return demonstration

def compose_input_topic_demonstration_1(task, demonstration_example_samples):

    demonstration = INSTRUCTION[task](example_label=True)

    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += "Input: " + row["text"] + "\n"
        demonstration += "Topic: " + row["topic"] + "\n"
        demonstration += "Label: " + label + "\n\n"

    return demonstration

def compose_stance_demonstration_1(task, demonstration_example_samples):

    demonstration = ""

    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += INSTRUCTION[task](example_label=True)
        demonstration += "Text: " + row["text"] + "\n"
        demonstration += "Target: " + row["target"] + "\n"
        demonstration += "Label: " + label + "\n\n"

    return demonstration


def fr_compose_x_stance_demonstration_1(task, demonstration_example_sample):

    demonstration = ""

    label = LABEL_MAPPING[task]["fr"][demonstration_example_sample["label"]]

    demonstration += INSTRUCTION["fr"][task](example_label=True)
    demonstration += "Texte: " + demonstration_example_sample["text"] + "\n"
    demonstration += "Sujet: " + demonstration_example_sample["topic"] + "\n"
    demonstration += "Label: " + label + "\n\n"

    return demonstration

def compose_x_stance_demonstration_1(task, demonstration_example_samples):

    demonstration = ""

    for i, row in demonstration_example_samples.iterrows():

        language = row["language"]

        if language == "de":
            demonstration += de_compose_x_stance_demonstration_1(task, row)
        elif language == "fr":
            demonstration += fr_compose_x_stance_demonstration_1(task, row)
        elif language == "it":
            demonstration += it_compose_x_stance_demonstration_1(task, row)

    return demonstration

def it_compose_x_stance_demonstration_1(task, demonstration_example_sample):

    demonstration = ""

    label = LABEL_MAPPING[task]["it"][demonstration_example_sample["label"]]

    demonstration += INSTRUCTION[task]["it"](example_label=True)
    demonstration += "Testo: " + demonstration_example_sample["text"] + "\n"
    demonstration += "Tema: " + demonstration_example_sample["topic"] + "\n"
    demonstration += "Label: " + label + "\n\n"

    return demonstration

def de_compose_x_stance_demonstration_1(task, demonstration_example_sample):

    demonstration = ""

    label = X_LABEL_MAPPING[task]["de"][demonstration_example_sample["label"]]

    demonstration += INSTRUCTION[task]["de"](example_label=True)
    demonstration += "Text: " + demonstration_example_sample["text"] + "\n"
    demonstration += "Thema: " + demonstration_example_sample["topic"] + "\n"
    demonstration += "Label: " + label + "\n\n"

    return demonstration

def compose_xnli_demonstration_1(task, demonstration_example_samples):

    demonstration = ""

    for i, row in demonstration_example_samples.iterrows():

        label = X_LABEL_MAPPING[task][row["label"]][row["language"]]

        demonstration += "Hypothesis: " + row["hypothesis"] + "\n"
        demonstration += "Premise: " + row["premise"] + "\n"
        demonstration += "Label: " + label + "\n\n"

    return demonstration


def compose_arg_topic_demonstration_1(task, demonstration_example_samples):

    demonstration = ""

    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += "Argument: " + row["text"] + "\n"
        demonstration += "Topic: " + row["topic"] + "\n"
        demonstration += "Label: " + label + "\n\n"

    return demonstration


def compose_evi_topic_demonstration_1(task, demonstration_example_samples):

    demonstration = ""

    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += INSTRUCTION[task](example_label=True)
        demonstration += "Evidence: " + row["text"] + "\n"
        demonstration += "Topic: " + row["topic"] + "\n"
        demonstration += "Label: " + label + "\n\n"

    return demonstration


def compose_pairwise_input_topic_demonstration_1(task, demonstration_example_samples):

    demonstration = ""

    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += "Input 1: " + row["text_1"] + "\n"
        demonstration += "Input 2: " + row["text_2"] + "\n"
        demonstration += "Topic: " + row["topic"] + "\n"
        demonstration += "Label: " + label + "\n\n"

    return demonstration

def compose_ukp_a_topic_demonstration_1(task, demonstration_example_samples):

    demonstration = ""

    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += INSTRUCTION[task](example_label=True)

        demonstration += "Argument 1: " + row["text_1"] + "\n"
        demonstration += "Argument 2: " + row["text_2"] + "\n"
        demonstration += "Topic: " + row["topic"] + "\n"
        demonstration += "Label: " + label + "\n\n"

    return demonstration


def compose_arg_q_topic_demonstration_1(task, demonstration_example_samples):

    demonstration = ""

    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += INSTRUCTION[task](example_label=True)

        demonstration += "Argument 1: " + row["text_1"] + "\n"
        demonstration += "Argument 2: " + row["text_2"] + "\n"
        demonstration += "Topic: " + row["topic"] + "\n"
        demonstration += "Label: " + label + "\n\n"

    return demonstration


def compose_input_demonstration_1(task, demonstration_example_samples):

    demonstration = ""
    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += "Input: " + row["text"] + "\n"
        demonstration += "Label: " +label + "\n\n"

    return demonstration

def compose_ag_news_demonstration_1(task, demonstration_example_samples):

    demonstration = ""
    for i, row in demonstration_example_samples.iterrows():

        label = LABEL_MAPPING[task][row["label"]]

        demonstration += "Text: " + row["text"] + "\n"
        demonstration += "Topic: " +label + "\n\n"

    return demonstration

def compose_input_topic_demonstration_query_1(row):
    instruction = "Input: " + row["text"] + "\n"
    instruction += "Topic: " + row["topic"] + "\n"
    instruction += "Label:"

    return instruction

def compose_stance_demonstration_query_1(row):
    instruction = "Text: " + row["text"] + "\n"
    instruction += "Target: " + row["target"] + "\n"
    instruction += "Label:"

    return instruction

def de_compose_x_stance_demonstration_query_1(row):
    instruction = "Text: " + row["text"] + "\n"
    instruction += "Thema: " + row["topic"] + "\n"
    instruction += "Label:"

    return instruction

def fr_compose_x_stance_demonstration_query_1(row):
    instruction = "Texte: " + row["text"] + "\n"
    instruction += "Sujet: " + row["topic"] + "\n"
    instruction += "Label:"

    return instruction

def it_compose_x_stance_demonstration_query_1(row):
    instruction = "Testo: " + row["text"] + "\n"
    instruction += "Tema: " + row["topic"] + "\n"
    instruction += "Label:"

    return instruction


def compose_review_demonstration_query_1(row):
    instruction = "Text: " + row["text"] + "\n"
    instruction += "Label:"

    return instruction


def compose_entailment_demonstration_query_1(row):
    instruction = "Text 1: " + row["text_1"] + "\n"
    instruction += "Text 2: " + row["text_2"] + "\n"
    instruction += "Label:"

    return instruction

def compose_xnli_demonstration_query_1(row):
    instruction = "Hypothesis: " + row["hypothesis"] + "\n"
    instruction += "Premise: " + row["premise"] + "\n"
    instruction += "Label:"

    return instruction

def compose_arg_topic_demonstration_query_1(row):
    instruction = "Argument: " + row["text"] + "\n"
    instruction += "Topic: " + row["topic"] + "\n"
    instruction += "Label:"

    return instruction

def compose_evi_topic_demonstration_query_1(row):
    instruction = "Evidence: " + row["text"] + "\n"
    instruction += "Topic: " + row["topic"] + "\n"
    instruction += "Label:"

    return instruction

def compose_input_demonstration_query_1(row):
    instruction = "Input: " + row["text"] + "\n"
    instruction += "Label:"

    return instruction

def compose_ag_news_demonstration_query_1(row):
    instruction = "Text: " + row["text"] + "\n"
    instruction += "Topic:"

    return instruction


def compose_instructions(example_samples, samples, task, seed, k, template_index, example_label=True, model_name="", scores=None):
    instruction_inputs = []

    random_state = seed


    for i, row in tqdm(samples.iterrows(), desc='compose instructions',):

        if "text" in row:
            other_samples = example_samples[example_samples["text"] != row["text"]]
        elif "premise" in row:
            other_samples = example_samples[
                (example_samples["hypothesis"] != row["hypothesis"]) & (example_samples["premise"] != row["premise"])
                ]
        else:
            other_samples = example_samples[
                (example_samples["text_1"] != row["text_1"]) & (example_samples["text_2"] != row["text_2"])
            ]

        if scores:
            top_indices = scores[i][-k*4:][::-1]
            top_indices = [top_index for top_index in top_indices if top_index in other_samples.index]

            try:
                instruction_example_samples = other_samples.loc[top_indices[:k]]
            except:
                print()
        else:
            instruction_example_samples = other_samples.sample(k, random_state=random_state)

        random_state = random_state + 1

        instruction_example_samples["label"] = instruction_example_samples["label"]

        if False:# "flan" in model_name:
            instruction_input = FLAN_DEMONSTRATION[task][template_index](task, instruction_example_samples)
            instruction_query = FLAN_DEMONSTRATION_QUERY[task][template_index](row)
        else:
            if k == 0:
                instruction_input = INSTRUCTION[task](example_label)
            else:
                instruction_input = DEMONSTRATION[task][template_index](task, instruction_example_samples)

            instruction_query = INSTRUCTION[task](example_label)
            instruction_query += DEMONSTRATION_QUERY[task][template_index](row)


        if "language" in row:
            input_example = InputExample(
                guid=i,
                text_a=instruction_input,
                text_b=instruction_query,
                label=row["label"],
                meta={
                    "language": row["language"]
                }
            )
        else:
            input_example = InputExample(
                guid=i,
                text_a=instruction_input,
                text_b=instruction_query,
                label=row["label"],
            )

        instruction_inputs.append(input_example)

    return instruction_inputs



def compose_ukp_a_demonstration_query_1(row):
    instruction = "Argument 1: " + row["text_1"] + "\n"
    instruction += "Argument 2: " + row["text_2"] + "\n"
    instruction += "Topic: " + row["topic"] + "\n"
    instruction += "Label:"

    return instruction

def compose_instructions_x(example_samples, samples, task, seed, k, template_index, example_label=True, model_name="", scores=None):
    instruction_inputs = []

    random_state = seed


    for i, row in tqdm(samples.iterrows(), desc='compose instructions',):

        if "text" in row:
            other_samples = example_samples[example_samples["text"] != row["text"]]
        elif "premise" in row:
            other_samples = example_samples[
                (example_samples["hypothesis"] != row["hypothesis"]) & (example_samples["premise"] != row["premise"])
                ]
        else:
            other_samples = example_samples[
                (example_samples["text_1"] != row["text_1"]) & (example_samples["text_2"] != row["text_2"])
                ]

        if scores:
            top_indices = scores[i][-k*4:][::-1]
            top_indices = [top_index for top_index in top_indices if top_index in other_samples.index]

            try:
                instruction_example_samples = other_samples.loc[top_indices[:k]]
            except:
                print()
        else:
            instruction_example_samples = other_samples.sample(k, random_state=random_state)

        random_state = random_state + 1

        instruction_example_samples["label"] = instruction_example_samples["label"]

        if False:# "flan" in model_name:
            instruction_input = FLAN_DEMONSTRATION[task][template_index](task, instruction_example_samples)
            instruction_query = FLAN_DEMONSTRATION_QUERY[task][template_index](row)
        else:
            if k == 0:
                instruction_input = INSTRUCTION[task](example_label)
            else:
                instruction_input = DEMONSTRATION[task][template_index](task, instruction_example_samples)

            instruction_query = INSTRUCTION[task][row["language"]](example_label)
            instruction_query += DEMONSTRATION_QUERY[task][template_index][row["language"]](row)


        if "language" in row:
            input_example = InputExample(
                guid=i,
                text_a=instruction_input,
                text_b=instruction_query,
                label=row["label"],
                meta={
                    "language": row["language"]
                }
            )
        else:
            input_example = InputExample(
                guid=i,
                text_a=instruction_input,
                text_b=instruction_query,
                label=row["label"],
            )

        instruction_inputs.append(input_example)

    return instruction_inputs

def get_scores_for_row(bm25, row):
    if "text" in row:
        tokenized_query = word_tokenize(row["text"])
    elif "premise" in row:
        tokenized_query = word_tokenize(row["hypothesis"] + " " + row["premise"])
    else:
        tokenized_query = word_tokenize(row["text_1"] + " " + row["text_2"])

    scores = bm25.get_scores(tokenized_query).argsort()[-100:][::-1]

    return scores


INSTRUCTION = {
    "arg-cls": ukp_argmin_instruction,
    "evi-cls": evi_sen_instruction,
    "arg-sim": ukp_a_instruction,
    "arg-qua": arg_q_instruction,
    "stance": stance_instruction,
    "review": review_instruction,
    "entail": entailment_instruction,
    "x-stance": {
        "de": de_x_stance_instruction,
        "fr": fr_x_stance_instruction,
        "it": it_x_stance_instruction,
    }
}


DEMONSTRATION = {
    "arg-cls": [
        compose_ukp_argmin_topic_demonstration_1
    ],
    "review": [
        compose_review_demonstration_1
    ],
    "entail": [
        compose_entailment_demonstration_1
    ],
    "stance": [
        compose_stance_demonstration_1
    ],
    "x-stance": [
        compose_x_stance_demonstration_1
    ],
    "evi-cls": [
        compose_evi_topic_demonstration_1
    ],
    "arg-sim": [
        compose_ukp_a_topic_demonstration_1
    ],
    "arg-qua": [
        compose_arg_q_topic_demonstration_1
    ],
}

DEMONSTRATION_QUERY = {
    "arg-cls": [
        compose_arg_topic_demonstration_query_1
    ],
    "stance": [
        compose_stance_demonstration_query_1
    ],
    "x-stance": [{
        "de": de_compose_x_stance_demonstration_query_1,
        "fr": fr_compose_x_stance_demonstration_query_1,
        "it": it_compose_x_stance_demonstration_query_1,
    }],
    "review": [
        compose_review_demonstration_query_1
    ],
    "entail": [
        compose_entailment_demonstration_query_1
    ],
    "evi-cls": [
        compose_evi_topic_demonstration_query_1
    ],
    "arg-sim": [
        compose_ukp_a_demonstration_query_1
    ],
    "arg-qua": [
        compose_ukp_a_demonstration_query_1
    ],
}
