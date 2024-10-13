# composition-miner
This repository contains the dataset and source code for _Large Language Models as a Tool for Mining Object Knowledge_.

## Content
This repository is organized as follows:
- `data/`:
  - `nounlist.txt`: The list of common physical objects used in experiments.
  - `csv/`: Contains the few-shot and zero-shot datasets presented in the paper, including preprocessed and clean versions.
  - `tsv/`: Includes raw and simplified human-annotated data
- `src/`: Contains all the source code used for data processing and generation.

## Reproduce the Data
To reproduce the data used in the experiments, follow the steps below.

## Noun List Preparation
There are two ways to get a list of nouns for well-perceived physical objects:
### Method 1: Use provided noun list
You can simply use the ``data/nounlist.txt`` file which contains the the list of nouns used in the paper.
### Method 2: Create your own noun list
You can follow the steps below to obtain your own list with necessary modifications.
#### 0. Install dependencies for creating a list of common physical nouns
``pip install -r prep_requirements.txt``
#### 1. Get a compressed file of Wikidata knowledgee base
Download the latest Wikidata dump (latest-all.json.bz2) from [Wikimedia Downloads](https://dumps.wikimedia.org/wikidatawiki/entities/). Because of the size of the data, this may take several hours to run.
#### 2. Extract relevant data
Run ``python src/wikiall.py latest-all.json.bz2`` to unzip the Wikidata file and extract the data of interst.
#### 3. Filter entries from Wikidata
Run ``python src/filterwiki.py`` to generate a csv file with filtered Wikidata entries
#### 4. Get further filtered list of nouns
Run ``python src/freqnouns.py filtered_wikidata.csv`` to get a list of physical objects that typical six-graders would know.

## Install Dependencies for Generating Object Composition Data
``pip install -r requirements.txt``

## Few-shot (In-context) Learning
``python src/fewshot.py``

## Zero-shot Multi-step Prompting
``python src/zeroshot.py``
