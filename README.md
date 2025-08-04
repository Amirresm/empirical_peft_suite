# Empirical PEFT Suite
This repository contains the replication package of our paper [Empirical Studies of Parameter Efficient Methods for Large Language Models of Code and Knowledge Transfer to R
](https://arxiv.org/pdf/2405.01553), along with supplementary tools used in our study. 

## Getting Started
The main replication code exists in the `suite` directory. Run the following in this context.
- Create a Python 3.11 environment (e.g., with [pyenv](https://github.com/pyenv/pyenv))
- Install dependencies `pip install -r requirements.txt`
- Run the main script `python main.py`
    - Refer to `suite/src/arg_utils.py` for an overview of arguments
    - Alternatively, use [Job Scripts](#job-scripts) as your starting scripts

## Job Scripts
Given the huge number of experiments in our study, we use a collection of bash scripts to manage parameters passed to the main script. This collection is a little finicky (this is a research repo after all), but should serve as a good reference and starting point to run different variations of these experiments.

> **Note**: You do not need to use these job scripts to execute tasks. These job scripts only serve as helpers to manage large numbers of scripts.

Job scripts exist in the `suite/jobs` directory. The directory structure **opinionated** and directly maps to the output of the jobs as `suite/jobs/{task}/{model}/{dataset}/{task_name}.sh`. We use structured task names as `e_{PEFTLIB}_{PEFT}.sh` to represent the task config where
- `PEFTLIB` is the PEFT library used ([AdapterHub](https://adapterhub.ml) `ah` or [Hugging Face PEFT](https://huggingface.co/docs/peft/en/index) `pl`)
- `PEFT` is the PEFT method (`lora`, `compacter` or `ia3`)

You can control shared configuration (such as dataset path, model path, etc...) per dataset, model and task type by editing each task's corresponding `data_vars.sh`, `model_vars.sh` and `job_vars.sh`, respectively.

Other configuration is controlled through
- `suite/jobs/local_vars.sh`, which controls the root path of the project, storage (where models and datasets are stored) and outputs,
- `suite/jobs/main_vars.sh`, which controls the system variables such as evaluation, logging, tokenization, generation, network and more (typically not changed much),
- `suite/jobs/main_sc.sh`, which sets up and runs the script,
- `suite/jobs/util_scripts/*`, which controls hyperparameters per model and dataset,
- and `suite/jobs/control_vars.sh`, which sets control variables (should be empty, unless you want to change some configuration quickly for testing).

Variables set through different scripts can override each other, with this order: `data_vars.sh` < `model_vars.sh` < `job_vars.sh` < `main_vars.sh` < `local_vars.sh` < `{task_name}.sh` < `control_vars.sh`.

> **Note**: Current presets expect a highly opinionated storage directory structure. Make sure to change the model and dataset path variables according to your directory structure.

## Tools
This repository also includes custom tools we used in various settings which are present in the `tools` directory. In summary, these tools are used for
- **Evaluate**: poss-execute evaluation (BLEU-4, ROUGE-*, CodeBLEU)
- **Explorer**: CLI results viewer
- **HumanEval**: HumanEval generation and evaluation utils (`pass@k` scores)
- **Migrate**: results migration
- **Present**: results transformation (to CSV, surveys, etc...)
- **Robustness**: dataset modification using GPT-4o, used in robustness experiments
- **Statistical Tests**: statistical tests

Note that these tools mostly depend on the opinionated directory structures defined before.

## Datasets
You can use the `jsonl` version of your own dataset of choosing with this code base. In this study (and to replicate), we use the `jsonl` version of the following datasets
- **spp_30k**: [Synthetic Python Problems(SPP) Dataset](https://huggingface.co/datasets/wuyetao/spp) | Python code generation
- **csn**: [CodeSearchNet](https://github.com/github/CodeSearchNet?tab=readme-ov-file#quickstart) | Python, Go, Java, JavaScript, PHP and Ruby for code summarization
- **MultiPL-T R**: R split of [MultiPL-T Fine-Tuning Datasets](https://huggingface.co/datasets/nuprl/MultiPL-T) | R code generation
- **R Code Summarization**: [Do Current Language Models Support Code Intelligence for R Programming Language?](https://zenodo.org/records/13871742) | R code summarization

Additionally, we use the following for code generation evaluation
- **HumanEval**: [human-eval](https://github.com/openai/human-eval) | Python code generation evaluation 
    - we use the implementation from [code-eval](https://github.com/abacaj/code-eval), with a clone present in this repository
- **MultiPL-E**: humaneval-r split of [MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E) | R code generation evaluation
    - The problem set dataset is present in [MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E)
    - Evaluation is carried out in a containerized environment using official docker images (refer to [BigCode](https://github.com/bigcode-project/bigcode-evaluation-harness?tab=readme-ov-file#docker-containers))

## Citation
If you find this repository useful, please cite our work using:
```BibTex
@misc{esmaeili2025empiricalstudiesparameterefficient,
      title={Empirical Studies of Parameter Efficient Methods for Large Language Models of Code and Knowledge Transfer to R}, 
      author={Amirreza Esmaeili and Iman Saberi and Fatemeh H. Fard},
      year={2025},
      eprint={2405.01553},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2405.01553}, 
}
```
