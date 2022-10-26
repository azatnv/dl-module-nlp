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
## Метрики

•	Precision (точность) - доля объектов, названных классификатором положительными и при этом действительно являющимися положительными.

$$ Precision = {TP \over (TP + FP)} $$

•	Recall (полнота) - доля объектов положительного класса, найденных классификатором, из всех объектов положительного класса.

$$ Recall = {TP \over (TP + FN)} $$

•	F-мера - гармоническое среднее между точностью и полнотой. Она стремится к нулю, если точность или полнота стремится к нулю.

$$ F = (\beta^{2}+1) * {Precision * Recall \over (\beta^{2}Precision + Recall)} $$

## Гиперпараметры

| hyperparameters | value |
| ------------------- | ------------------- |
| Batch Size | classifier: 16, embedder: 32 |
| Epochs | classifier: 1, embedder: 1 |
| Optimizer | torch.optim.AdamW(lr=1e-4, eps=1e-6) |
| Warmup Steps | 5% batches ~ 1k |
| Scheduler | WarmupLinear |

## Результаты работы

•	Сравение методов поиска
| method  | speed |
| ------------- | ------------- |
| kNN | 230ms  |
| Approximate NN | 180ms  |

•	Тестовые метрики
| Precission  | Recall | F-мера |
| ------------- | ------------- | ------------- |
| 0.98 | 0.18 | 0.31 |

## Pipeline
<img width="603" alt="image" src="https://user-images.githubusercontent.com/31849841/198042622-9ca7764b-07b6-4eec-a1fa-bcff9d760922.png">

## Example
![image](https://user-images.githubusercontent.com/31849841/198023783-3569d01a-8dc6-47b6-88d8-77a25ad214ac.png)
