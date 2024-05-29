# How to Handle Different Types of Out-of-Distribution Scenarios in Computational Argumentation? A Comprehensive and Fine-Grained Field Study

This repository includes the code of the experiments reported in [How to Handle Different Types of Out-of-Distribution Scenarios in Computational Argumentation? A Comprehensive and Fine-Grained Field Study
](https://arxiv.org/abs/2309.08316).


> **Abstract:** The advent of pre-trained Language Models (LMs) has markedly advanced natural language processing, but their efficacy in out-of-distribution (OOD) scenarios remains a significant challenge. Computational argumentation (CA), modeling human argumentation processes, is a field notably impacted by these challenges because complex annotation schemes and high annotation costs naturally lead to resources barely covering the multiplicity of available text sources and topics. Due to this data scarcity, generalization to data from uncovered covariant distributions is a common challenge for CA tasks like stance detection or argument classification. This work systematically assesses LMs' capabilities for such OOD scenarios. While previous work targets specific OOD types like topic shifts or OOD uniformly, we address three prevalent OOD scenarios in CA: topic shift, domain shift, and language shift. Our findings challenge the previously asserted general superiority of in-context learning (ICL) for OOD. We find that the efficacy of such learning paradigms varies with the type of OOD. Specifically, while ICL excels for domain shifts, prompt-based fine-tuning surpasses for topic shifts. To sum up, we navigate the heterogeneity of OOD scenarios in CA and empirically underscore the potential of base-sized LMs in overcoming these challenges.

Contact person: Andreas Waldis, andreas.waldis@live.com

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to e-mail us or report an issue if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.


## Project structure

* `src/` -- necessary code to run the experiments
* `task/` -- contains the out-of-distribution (ood) and in-distribution (id) task files
  * ct: cross-topic (arg-qua, arg-sim, arg-cls, evi-cls, x-stance)
  * cd: cross-domain (review, stance, entail, x-review)
  * cl: cross-language (x-stance, x-review)
  * it: in-topic (arg-qua, arg-sim, arg-cls, evi-cls, x-stance)
  * id: in-domain (review, stance, entail, x-review)
  * il: in-lanugage (x-stance, x-review)

## Setup

This repository requires Python3.6 or higher; further requirements can be found in the requirements.txt. 
Install them with the following command:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Next, you need to setup the `.env`.
Either copy `.env_dev` (development) or `.env_prod` (production) to `.env` and set your openai (`OPENAI_KEY`) key, if you would like to run the in-context learning experiments. 

```
$ cp .env_dev .env #development
$ cp .env_prod .env #production
```

Finally, you need to log in with your wandb account for performance reporting. 

```
$ wandb login
```

## Tasks

This work relies on the following different datasets:
*  Argument Quality (`arg-qua`), available [here](https://research.ibm.com/haifa/dept/vst/debating_data.shtml)
*  Argument Similarity (`arg-sim`), available [here](https://huggingface.co/datasets/UKPLab/UKP_ASPECT)
*  Argument Classification (`arg-cls`), available [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345)
*  Evidence Classification (`evi-cls`), available [here](https://research.ibm.com/haifa/dept/vst/debating_data.shtml)
*  Entailment (`entail`), RTE is available [here](https://huggingface.co/datasets/nyu-mll/glue), SCITAIL [here](https://huggingface.co/datasets/allenai/scitail), and HANS [here](https://huggingface.co/datasets/jhu-cogsci/hans)
*  Sentiment Analysis (`review`), available [here](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)
*  Stance Detection (`stance`), SemEval2016Task6 is available [here](http://alt.qcri.org/semeval2016/task6/), EMERGENT [here](https://github.com/willferreira/mscproject), and IAC [here](https://nlds.soe.ucsc.edu/iac)
*  Multi-Lingual Stance Detection (`x-stance`), available [here](https://huggingface.co/datasets/ZurichNLP/x_stance)
*  Multi-Lingual Sentiment Analysis (`x-review`), available [here](https://zenodo.org/record/3251672/files/cls-acl10-unprocessed.tar.gz)

Some of these datasets requires accepting conditions, but we are happy to share our parsed version with you. 
Send us these datasets as zip to proof that you have access.
Afterward, we share our splits and you can put them into the `tasks` folder. 

## Running the experiments

Please use the following scripts to run the different experiments.
* `run_frozen.py`, mono- and multi-lingual linear probing
* `run_prompt.py`, mono-lingual prompting
* `run_x_prompt.py`, multi-lingual prompting
* `run_fine-tuning.py`, mono- and multi-lingual fine-tuning
* `run_frozen_single_fix_step.py`, `run_fine-tuning_pre-initialized.py`, probing-first fine-tuning afterwards
* `run_prompt_tuning.py`, mono-lingual prompt-tuning
* `run_x_prompt_tuning.py`, multi-lingual prompt-tuning
* `run_peft_prompt_tuning.py`, lora mono-lingual prompt-tuning
* `run_icl.py`, mono-lingual in prompting

## Citation

```
@inproceedings{Waldis2023HowTH,
  title={How to Handle Different Types of Out-of-Distribution Scenarios in Computational Argumentation? A Comprehensive and Fine-Grained Field Study},
  author={Andreas Waldis and Iryna Gurevych},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:262013145}
}
```
