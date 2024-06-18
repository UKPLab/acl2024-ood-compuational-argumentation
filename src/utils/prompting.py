import itertools
import string
from collections import Counter
from typing import List, Optional

import openai
import tiktoken
import torch
import torch.nn.functional as F
from line_profiler_pycharm import profile
from openprompt.data_utils import InputFeatures
from openprompt.plms import ModelClass, MLMTokenizerWrapper, _MODEL_CLASSES, T5TokenizerWrapper, LMTokenizerWrapper
from openprompt.prompts import AutomaticVerbalizer, ManualVerbalizer, SoftVerbalizer
from retry import retry
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration, ElectraConfig, ElectraTokenizer, \
    ElectraForMaskedLM, DebertaConfig, DebertaTokenizer, DebertaForMaskedLM, MT5Config, MT5Tokenizer, \
    MT5ForConditionalGeneration, XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForMaskedLM, DebertaV2Config, \
    DebertaV2Tokenizer, DebertaV2ForMaskedLM, GPTNeoXForCausalLM, GPTNeoXTokenizerFast, GPTNeoXConfig, \
    OPTConfig, OPTForCausalLM, GPT2Tokenizer, LlamaConfig, LlamaTokenizer, LlamaForCausalLM, GPTJConfig, \
    GPTJForCausalLM, BloomConfig, BloomForCausalLM, BloomTokenizerFast

X_TEMPLATES = {
    "x-stance":{
        "de":'Die Haltung von {"placeholder": "text_a"} ist {"mask"} zu {"placeholder": "text_b"}.',
        "fr":'L\'attitude de {"placeholder": "text_a"} est {"mask"} envers {"placeholder": "text_b"}.',
        "it":'L\'atteggiamento di {"placeholder": "text_a"} è {"mask"} verso {"placeholder": "text_b"}.',
    },
    "x-sentiment":{
        "en": 'The sentiment of {"placeholder": "text_a", "shortenable": "True"} is {"mask"}',
        "de": 'Die Stimmung von {"placeholder": "text_a", "shortenable": "True"} ist {"mask"}',
        "fr": 'Le sentiment de {"placeholder": "text_a", "shortenable": "True"} est {"mask"}',
        "jp": '{"placeholder": "text_a", "shortenable": "True"} の感情は {"mask"} です',
    }
}
TEMPLATES = {
    "entail": [
        '{"placeholder": "text_a"}? {"mask"}, {"placeholder": "text_b"}.'
    ],
    "arg-cls": [
        'The stance that {"placeholder": "text_a"} is {"mask"} on {"placeholder": "text_b"}.',
        '{"placeholder": "text_a"} is {"mask"} for {"placeholder": "text_b"}.',
        '{"placeholder": "text_a"} {"placeholder": "text_b"} {"mask"}',
        'The attitude of {"placeholder": "text_a"} is {"mask"} regarding {"placeholder": "text_b"}.',
    ],
    "review": [
        'The sentiment of {"placeholder": "text_a"} is {"mask"}.',
    ],
    "temporal-review": [
        'The sentiment of {"placeholder": "text_a"} is {"mask"}.',
    ],
    "stance":[
        'The stance that {"placeholder": "text_a"} is {"mask"} on {"placeholder": "text_b"}.',
        '{"placeholder": "text_a"} is {"mask"} for {"placeholder": "text_b"}.',
        '{"placeholder": "text_a"} {"placeholder": "text_b"} {"mask"}',
        'The attitude of {"placeholder": "text_a"} is {"mask"} regarding {"placeholder": "text_b"}.',
    ],
    "evi-cls": [
        '{"placeholder": "text_a"} is {"mask"} evidence regarding {"placeholder": "text_b"}.',
        'Evidence {"placeholder": "text_a"} is {"mask"} evidence regarding topic {"placeholder": "text_b"}.',
        '{"placeholder": "text_a"} {"placeholder": "text_b"} {"mask"}',
        'Is {"placeholder": "text_a"} an evidence regarding {"placeholder": "text_b"}? {"mask"}',
    ],
    "arg-sim": [
        '{"placeholder": "text_a"} is {"mask"} to {"placeholder": "text_b"} regarding {"meta": "topic"}.',
        'Argument {"placeholder": "text_a"} is {"mask"} to argument {"placeholder": "text_b"} regarding topic {"meta": "topic"}.',
        '{"placeholder": "text_a"} {"placeholder": "text_b"} {"meta": "topic"} {"mask"}',
    ],
    "arg-qua": [
        '{"placeholder": "text_a"} is {"mask"} than {"placeholder": "text_b"} regarding {"meta": "topic"}.',
        'Argument {"placeholder": "text_a"} is {"mask"} than argument {"placeholder": "text_b"} regarding topic {"meta": "topic"}.',
        '{"placeholder": "text_a"} {"placeholder": "text_b"} {"meta": "topic"} {"mask"}',
    ],
}


X_STATIC_VERBALIZING = {
    "x-stance" : {
        0 : [
            "pro",
            "dafür"
            "Par",
            "per questo",
            "per",
            "pour ça"
        ],
        1 : [
            "Kontra",
            "dagegen",
            "contro di esso",
            "contro",
            "contre",
            "encontre"
        ],
    },
    "x-sentiment" : {
        0 : [
            "negative",
            "negativ"
            "négative",
            "ネガティブ",
        ],
        1 : [
            "positive",
            "positiv",
            "ポジティブ",
        ],
    },
}
STATIC_VERBALIZING = {
    "arg-cls" : {
        0 : ["neutral", "unrelated"], #Neutral
        1 : ["for", "in favor", "support"], #For
        2 : ["against", "versus"], #Against
    },
    "arg-sim" : {
        0 : ["similar"],
        1 : ["different"],
    },
    "entailment" : {
        0 : ["yes"],
        1 : ["no"],
    },
    "entail" : {
        0 : ["yes"],
        1 : ["no"],
    },
    "arg-qua" : {
        0 : ["better"],
        1 : ["worse"],
    },
    "evi-cls" : {
        0 : ["no"],
        1 : ["yes"],
    },
    "review" : {
        0 : ["negative"],
        1 : ["positive"],
    },
    "stance": {
        0: ["pro", "favor", "for"],
        1: ["anti", "against"],
        2: ["other", "none", "observing"],
    },
}


def get_model_family(model_name):
    if "albert" in model_name.lower():
        return "albert"
    elif "gpt-j" in model_name.lower():
        return "gpt-j"
    elif "bloom" in model_name.lower():
        return "bloom"
    elif "llama" in model_name.lower():
        return "llama"
    elif "xlm-roberta" in model_name.lower():
        return "xlm-roberta"
    elif "roberta" in model_name.lower():
        return "roberta"
    elif "deberta-v2" in model_name.lower():
        return "deberta-v2"
    elif "gpt-neox" in model_name.lower():
        return "gpt-neox"
    elif "deberta-v3" in model_name.lower():
        return "deberta-v2"
    elif "deberta" in model_name.lower():
        return "deberta"
    elif "bert-mlm" in model_name.lower():
        return "roberta"
    elif "bert" in model_name.lower():
        return "bert"
    elif "mt5" in model_name.lower() or "mt0" in model_name.lower():
        return "mt5"
    elif "t5" in model_name.lower() or "t0" in model_name.lower() or "tk-instruct" in model_name.lower() or "flan" in model_name.lower():
        return "t5"
    elif "t5-lm" in model_name.lower():
        return "t5-lm"
    elif "bart" in model_name.lower():
        return "bart"
    elif "electra" in model_name.lower():
        return "electra"
    elif "opt" in model_name.lower():
        return "opt"
    elif "gpt2" in model_name.lower():
        return "gpt2"
    elif "gpt" in model_name.lower():
        return "gpt"
    elif "llama" in model_name.lower():
        return "llama"
    elif "vicuna" in model_name.lower():
        return "llama"
    elif "pythia" in model_name.lower():
        return "gpt-neox"

def get_prompt_preds(data_loader, prompt_model, use_cuda=False, remove_loss_id=False):

    all_labels = []
    all_preds = []

    for inputs in tqdm(data_loader):
        if use_cuda:
            inputs = inputs.cuda()

        if remove_loss_id and "loss_ids" in inputs:
            del inputs["loss_ids"]

        logits = prompt_model(inputs)
        labels = inputs['label']
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    return all_preds

@retry(Exception, tries=5, backoff=2, delay=1)
def run_gpt_prompt(prompt, model_name):
    return openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1,
        #logit_bias=token_id_bias
    )

def get_gpt_prompt_preds(instructions, token_mappings, model_name="gpt-3.5-turbo"):

    all_preds = []

    enc = tiktoken.encoding_for_model(model_name)

    tokens = set(token_mappings.keys())

    unique_labels = list(set([ele.label for ele in instructions]))

    pred_tokens = []

    for i, instruction in tqdm(enumerate(instructions)):
        label = instruction.label
        response = run_gpt_prompt(instruction.text_a + instruction.text_b, model_name)
        pred_token = response["choices"][0]["message"]["content"].lower().strip().translate(str.maketrans('', '', string.punctuation))
        if pred_token in token_mappings:
            pred = token_mappings[pred_token]
        else:
            others = [ele for ele in unique_labels if ele != label]
            pred = others[0]

        all_preds.append(pred)
        pred_tokens.append(pred_token)

    return all_preds, pred_tokens


def get_gpt_prompt_preds_x(instructions, token_mappings, model_name="gpt-3.5-turbo"):

    all_preds = []

    enc = tiktoken.encoding_for_model(model_name)

    tokens = set(token_mappings.keys())

    unique_labels = list(set([ele.label for ele in instructions]))

    pred_tokens = []

    for i, instruction in tqdm(enumerate(instructions)):
        label = instruction.label
        language = instruction.meta["language"]
        response = run_gpt_prompt(instruction.text_a + instruction.text_b, model_name)
        pred_token = response["choices"][0]["message"]["content"].lower().strip().translate(str.maketrans('', '', string.punctuation))
        if pred_token in token_mappings:
            pred = token_mappings[pred_token][language]
        else:
            others = [ele for ele in unique_labels if ele != label]
            pred = others[0]

        all_preds.append(pred)
        pred_tokens.append(pred_token)

    return all_preds, pred_tokens



def new_collate_fct(batch: List):
    r'''
    This function is used to collate the input_features.

    Args:
        batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.

    Returns:
        :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
    '''


    elem = batch[0]
    return_dict = {}

    if "input_ids_len" in batch[0]:
        max_length = max([d["input_ids_len"] for d in batch])
    else:
        pad_token = Counter([int(ele) for ele in itertools.chain.from_iterable([d["input_ids"][-3:].detach().cpu().numpy() for d in batch])]).most_common(1)[0][0]
        max_length = max([(d["input_ids"] != pad_token).sum() for d in batch])

    for key in elem:
        if key == "encoded_tgt_text":
            return_dict[key] = [d[key] for d in batch]
        else:

            try:
                if key in ["input_ids", "attention_mask", "token_type_ids", "loss_ids"]:
                    return_dict[key] = torch.stack([d[key][:max_length] for d in batch])
                else:
                    return_dict[key] = default_collate([d[key] for d in batch])
            except:
                print(f"key{key}\n d {[batch[i][key] for i in range(len(batch))]} ")

    return InputFeatures(**return_dict)

def new_soft_collate_fct(batch: List):
    r'''
    This function is used to collate the input_features.

    Args:
        batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.

    Returns:
        :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
    '''


    elem = batch[0]
    return_dict = {}

    max_length = max([torch.nonzero(d["input_ids"]).max() for d in batch])

    for key in elem:
        if key == "encoded_tgt_text":
            return_dict[key] = [d[key] for d in batch]
        else:

            try:
                if key in ["input_ids", "attention_mask", "token_type_ids"]:
                    return_dict[key] = torch.stack([d[key][:max_length] for d in batch])
                else:
                    return_dict[key] = default_collate([d[key] for d in batch])
            except:
                print(f"key{key}\n d {[batch[i][key] for i in range(len(batch))]} ")

    return InputFeatures(**return_dict)



class CPUAutomaticVerbalizer(AutomaticVerbalizer):
    @profile
    def register_buffer(self, logits, labels):
        r'''

        Args:
            logits (:obj:`torch.Tensor`):
            labels (:obj:`List`):
        '''

        logits = F.softmax(logits.detach().cpu(),dim=-1)
        labels = labels.detach().cpu()
        if self.probs_buffer is None :
            self.probs_buffer = [logits]
            self.labels_buffer = [labels]
        else:
            self.probs_buffer.append(logits)
            self.labels_buffer.append(labels)

    @profile
    def _find_verbalizer(self, words_per_label: int = 1, num_candidates: int = 1000, balance: bool = True,
                                 score_fct: str = 'llr'):

        probs = torch.vstack(self.probs_buffer)
        labels = torch.hstack(self.labels_buffer)
        candidates = self._get_candidates(num_candidates=num_candidates, probs=probs, labels=labels)
        label_words =  self._get_top_words(probs=probs, labels=labels, candidates=candidates, balance=balance, words_per_label=words_per_label,
                                           score_fct=score_fct)
        return label_words

    def _get_top_words(self,
                       probs: torch.Tensor,
                       labels: torch.Tensor,
                       candidates: List[torch.Tensor],
                       balance: bool = True,
                       words_per_label: int = 10,
                       score_fct: Optional[str] = 'llr'):
        label_words_ids = []
        for label_id in range(self.num_classes):
            label_mask = (labels==label_id).to(torch.float)
            probs_per_label = probs[:, candidates[label_id]]
            if score_fct == 'llr':
                s = self._log_likelihood_ratio(probs_per_label, label_mask, balance)
            elif score_fct == 'ce':
                s = self._cross_entropy(probs_per_label, label_mask, balance)
            else:
                raise ValueError(f"Score function '{score_fct}' not implemented")
            sorted_ids  = torch.argsort(s, descending=True)[:words_per_label]
            selected_ids = candidates[label_id][sorted_ids]
            label_words_ids.append(selected_ids)
        label_words_ids = torch.vstack(label_words_ids)
        return label_words_ids


class CPUAutomaticVerbalizer16(AutomaticVerbalizer):

    def register_buffer(self, logits, labels):
        r'''

        Args:
            logits (:obj:`torch.Tensor`):
            labels (:obj:`List`):
        '''

        logits = F.softmax(logits.float().detach().cpu(),dim=-1)
        labels = labels.detach().cpu()
        if self.probs_buffer is None :
            self.probs_buffer = logits
            self.labels_buffer = labels
        else:
            self.probs_buffer = torch.vstack([self.probs_buffer, logits])
            self.labels_buffer = torch.hstack([self.labels_buffer, labels])


class ManualVerbalizer16(ManualVerbalizer):
    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.float().reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

class SoftVerbalizer16(SoftVerbalizer):
    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.float().reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

def update_plms():
    _MODEL_CLASSES["bart"] = ModelClass(**{
        'config': BartConfig,
        'tokenizer': BartTokenizer,
        'model':BartForConditionalGeneration,
        'wrapper': MLMTokenizerWrapper,
    })
    _MODEL_CLASSES["xlm-roberta"] = ModelClass(**{
        'config': XLMRobertaConfig,
        'tokenizer': XLMRobertaTokenizer,
        'model':XLMRobertaForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    })
    _MODEL_CLASSES["electra"] = ModelClass(**{
        'config': ElectraConfig,
        'tokenizer': ElectraTokenizer,
        'model':ElectraForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    })
    _MODEL_CLASSES["deberta"] = ModelClass(**{
        'config': DebertaConfig,
        'tokenizer': DebertaTokenizer,
        'model':DebertaForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    })

    _MODEL_CLASSES["deberta-v2"] = ModelClass(**{
        'config': DebertaV2Config,
        'tokenizer': DebertaV2Tokenizer,
        'model':DebertaV2ForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    })
    _MODEL_CLASSES["mt5"] = ModelClass(**{
        'config': MT5Config,
        'tokenizer': MT5Tokenizer,
        'model': MT5ForConditionalGeneration,
        'wrapper': T5TokenizerWrapper
    })
    _MODEL_CLASSES["gpt-neox"] = ModelClass(**{
        'config': GPTNeoXConfig,
        'tokenizer': GPTNeoXTokenizerFast,
        'model': GPTNeoXForCausalLM,
        'wrapper': LMTokenizerWrapper
    })
    _MODEL_CLASSES["gpt-j"] = ModelClass(**{
        'config': GPTJConfig,
        'tokenizer': GPT2Tokenizer,
        'model': GPTJForCausalLM,
        'wrapper': LMTokenizerWrapper
    })
    _MODEL_CLASSES["opt"] = ModelClass(**{
        'config': OPTConfig,
        'tokenizer': GPT2Tokenizer,
        'model': OPTForCausalLM,
        'wrapper': LMTokenizerWrapper
    })
    _MODEL_CLASSES["bloom"] = ModelClass(**{
        'config': BloomConfig,
        'tokenizer': BloomTokenizerFast,
        'model': BloomForCausalLM,
        'wrapper': LMTokenizerWrapper
    })
    _MODEL_CLASSES["llama"] = ModelClass(**{
        'config': LlamaConfig,
        'tokenizer': LlamaTokenizer,
        'model': LlamaForCausalLM,
        'wrapper': LMTokenizerWrapper
    })

