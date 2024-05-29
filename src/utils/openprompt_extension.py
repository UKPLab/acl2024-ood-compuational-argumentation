import warnings

import numpy
import numpy as np
import transformers
from openprompt.plms import get_model_class


def add_special_tokens(self, encoder_inputs):
    # add special tokens

    if any(numpy.array(encoder_inputs["loss_ids"]) > 10):
        print(encoder_inputs["loss_ids"])

    for key in encoder_inputs:
        if key == "input_ids":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                encoder_inputs[key] = self.tokenizer.build_inputs_with_special_tokens(
                    encoder_inputs[key])
        else:
            if "Fast" in type(self.tokenizer).__name__:
                special_tokens_mask = np.array([0] * len(encoder_inputs[key]))
            else:
                special_tokens_mask = np.array(self.tokenizer.get_special_tokens_mask(encoder_inputs[key]))

            with_special_tokens = np.array(self.tokenizer.build_inputs_with_special_tokens(encoder_inputs[key]))
            if key in ["soft_token_ids"]: # TODO maybe more than this
                encoder_inputs[key] =  ((1-special_tokens_mask) * with_special_tokens).tolist() # use 0 as special
            else:
                encoder_inputs[key] =  ((1-special_tokens_mask) * with_special_tokens - special_tokens_mask*100).tolist() # use -100 as special

    return encoder_inputs


def add_plain_special_tokens(self, encoder_inputs):
    # add special tokens

    if any(numpy.array(encoder_inputs["loss_ids"]) > 10):
        print(encoder_inputs["loss_ids"])

    for key in encoder_inputs:
        if key == "input_ids":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                encoder_inputs[key] = self.tokenizer.build_inputs_with_special_tokens(
                    encoder_inputs[key])


    return encoder_inputs



def load_plm(model_name, model_path, specials_to_add = None):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`wrapper`: The wrapper class of this plm.
    """
    model_class = get_model_class(plm_type = model_name)
    model_config = model_class.config.from_pretrained(model_path)
    # you can change huggingface model_config here
    # if 't5'  in model_name: # remove dropout according to PPT~\ref{}
    #     model_config.dropout_rate = 0.0
    if 'gpt' in model_name: # add pad token for gpt
        specials_to_add = ["<pad>"]
        # model_config.attn_pdrop = 0.0
        # model_config.resid_pdrop = 0.0
        # model_config.embd_pdrop = 0.0
    model = transformers.AutoModel.from_pretrained(model_path, config=model_config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    wrapper = model_class.wrapper


    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)

    if 'opt' in model_name:
        tokenizer.add_bos_token=False
    return model, tokenizer, model_config, wrapper
