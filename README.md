# KGC_LLM
This project contains multiple datasets and their corresponding evaluation and running scripts. Details are as follows:

## Directory Structure

- **Codex**: Contains the Codex dataset.
- **WN18RR**: Contains the WN18RR dataset.
- **other**: Contains other datasets, including Yago3-10.

Datasets source: [datasets_knowledge_embedding](https://github.com/villmow/datasets_knowledge_embedding), [codex](https://github.com/tsafavi/codex).

## File Descriptions

### dataset.py

Defines classes for constructing different datasets for instantiation.

### evaluate

Contains evaluations of the results for the WN18RR and Yago3-10 datasets.

### evaluate_codex

Contains evaluations of the results for the Codex dataset.

### result

Contains running scripts for the corresponding datasets and the results of 1000 samples with a random seed of 42.

- **run.py**: The running script. The `zero_shot_list` and `few_shot_list` denote lists of experiment names and will automatically execute all experiments listed. The `gpt4` parameter indicates whether to use GPT-4 as the output model, defaulting to GPT-3.5.
- **evaluate.ipynb**: The notebook file for analyzing results.

### utils.py

Contains all utility classes. Before running the experiments, replace the `api` values in the `get_response` and `get_response_gpt4` methods with your own OpenAI API key.

## Usage Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Replace the OpenAI API key:
    In the `utils.py` file, replace the `api` values in the `get_response` and `get_response_gpt4` methods with your own OpenAI API key.

4. Run the experiments:
    ```bash
    python result/dataset/run.py
    ```
