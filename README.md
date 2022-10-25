# Company name duplicate search

## Setup
```Linux Kernel Module
conda create -n env_name --python=3.10

conda activate env_name

pip install -r requirements.txt
```
## Usage
```
python main.py
```
## Evaluation on test dataset (time-consuming)
```
python ./test/test.py [--file ./test/companies_test.csv ]
```