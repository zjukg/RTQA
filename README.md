# RTQA

## 🚀 Overview
<p align="center">
  <img src="figure/main.png" alt="RTQA pipeline" width="700"/>
</p>

This repository contains the code and resources for the RTQA framework, as described in the paper: *[RTQA: Recursive Thinking for Complex Temporal Knowledge Graph Question Answering with Large Language Models](https://arxiv.org/abs/2509.03995)*.


## 🔔 News
- [x] Our paper has been accepted to **EMNLP 2025 main** 🎉
- [x] Release the code and resources before **2025-09-30**
- [x] Our paper is released on arxiv ! 


## 🛠️ Setting Up 


```bash
git clone https://github.com/zjukg/RTQA.git
conda create -n RTQA python=3.9.21
conda activate RTQA
pip install -r requirements.txt
```


## 📊 Obtaining Datasets

The RTQA framework uses the **MultiTQ** and **TimelineKGQA** datasets for evaluation. Below are instructions to download these datasets:

### MultiTQ Dataset

1. Visit the [https://github.com/czy1999/MultiTQ](https://github.com/czy1999/MultiTQ).

```bash
git clone https://github.com/czy1999/MultiTQ.git
cd MultiTQ/data
unzip Dataset.zip
```

2. Alternatively, download the dataset directly from Hugging Face:

🤗Datasets Link: https://huggingface.co/datasets/chenziyang/MultiTQ

### TimelineKGQA Dataset

Visit the [https://github.com/PascalSun/TimelineKGQA/tree/main/Datasets](https://github.com/PascalSun/TimelineKGQA/tree/main/Datasets)

```bash
git clone https://github.com/PascalSun/TimelineKGQA.git
cd Datasets
```
*Note*: The TimelineKGQA dataset is generated based on ICEWS Actor and CronQuestions KG. We only use the CronQuestions KG part.

## 📕 Evaluation
```bash
cd MultiTQ/TimelineKGQA
cd TemQuesDecom
python 0_get_prompt.py
phthon 1_query.py
python 2_combine.py
python 3_self_check.py
python 4_postprocess.py
phthon 5_postprocess_tree.py
cd ../RecursiveSolver
python 1_built_tree_time.py
python 2_run.py
python 3_get_f1.py
```

## 🤝 Cite:
Please consider citing this paper if you find our work useful.

```bash
@misc{gong2025rtqarecursivethinking,
      title={RTQA : Recursive Thinking for Complex Temporal Knowledge Graph Question Answering with Large Language Models}, 
      author={Zhaoyan Gong and Juan Li and Zhiqiang Liu and Lei Liang and Huajun Chen and Wen Zhang},
      year={2025},
      eprint={2509.03995},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.03995}, 
}
```
