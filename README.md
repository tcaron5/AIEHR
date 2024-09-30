# AI-EHR

## Team Members
- Suzanne Wendelken (s.wendelken@northeastern.edu)
- Anson Antony
- Dushyant Mahajan
- Troy Caron
- Walid Saba
- Jimmy Shanahan
- Rajashekar Korutla

We are a multidisciplinary team with backgrounds in machine learning, healthcare, and technology, collaborating on the **Discharge Me! Task**. Our goal is to create a model that generates accurate discharge summaries from clinical data, using the MIMIC-IV dataset. This project aims to enhance the automation of discharge instructions, improving the efficiency of clinical documentation.

## Project Overview
The **Discharge Me!** task is part of an initiative to advance text generation in healthcare. The objective of this project is to reduce the time and effort clinicians spend on writing detailed notes in the electronic health record (EHR). Clinicians play a crucial role in documenting patient progress in discharge summaries, but the creation of concise yet comprehensive hospital course summaries and discharge instructions often demands a significant amount of time. This can lead to clinician burnout and operational inefficiencies within hospital workflows. By streamlining the generation of these sections, we can not only enhance the accuracy and completeness of clinical documentation but also significantly reduce the time clinicians spend on administrative tasks, ultimately improving patient care quality.

More information can be found at [Discharge Me](https://stanford-aimi.github.io/discharge-me/)


## MIMIC Data
We use the **MIMIC-IV** dataset, which includes de-identified health-related data from thousands of hospital admissions. This dataset provides detailed information about patient admissions, treatments, and outcomes, and is crucial for training our models to understand clinical contexts and generate relevant discharge summaries.

To gain access to the [MIMIC-IV](https://physionet.org/content/mimiciv/3.0/) dataset, complete training

1. [https://physionet.org/content/mimiciii/view-required-training/1.4/#1](https://physionet.org/content/mimiciii/view-required-training/1.4/#1)
2. [https://physionet.org/about/citi-course/](https://physionet.org/about/citi-course/)


## Relevant Publications

- **Conference Paper**: ["Exploring Transformer-Based Approaches for Clinical Text Summarization"](https://aclanthology.org/2024.bionlp-1.63/)
- ["Learning to Generate Clinical Discharge Summaries Using Reinforcement Learning"](https://arxiv.org/abs/2401.01469)

These papers provide insights into the state-of-the-art methods we explore in our project.


## Contribution Guidelines

To maintain consistency and collaboration across our team, please follow these practices when editing and contributing to this repository:

### 1. Branching
- Always work on a **new branch** for any changes. Name your branch based on the feature, fix, or task you are addressing.
- Use the following command to create a new branch:
  ```bash
  git checkout -b your-branch-name
  
### 2. After making changes
- Write clear and concise **commit messages** that describe the changes you've made.
  ```bash
  git add .
  git commit -m "Your descriptive commit message"
### 3. Push the branch:
  ```bash
  git push origin your-branch-name
### 4. Creating a pull request:
- Go to the repository on github.
- Before submitting a pull request, ensure your changes are tested and documented.
- Create a pull request from your-branch-name into main.
- Once approved, click Merge Pull Request
### 5. Sync your local main:
  ```bash
  git checkout main
  git pull origin main
### 6. Delete the branch (optional):
  ```bash
  git branch -d your-branch-name
