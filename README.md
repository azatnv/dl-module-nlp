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
$$ Precision = TP / (TP + FP) $$

•	Recall (полнота) - доля объектов положительного класса, найденных классификатором, из всех объектов положительного класса.  
$$ Recall = TP / (TP + FN) $$

•	F-мера - гармоническое среднее между точностью и полнотой. Она стремится к нулю, если точность или полнота стремится к нулю.
$$ F = (\beta^{2}+1) * (Precision * Recall / (\beta^{2}Precision + Recall)) $$

## Гиперпараметры

| hyperparameters | value |
| ------------------- | ------------------- |
| Batch Size | classificator: 16, embedder: 32 |
| Epochs | classificator: 16\n embedder: 32 |
| Optimizer | torch.optim.AdamW(lr=1e-4, eps=1e-6) |
| Warmup Steps | 5% batches ~ 1k |
| Scheduler | WarmupLinear |

## Результаты работы

#### Скорость поиска

Метод поиска по косинусному сходству

| method  | speed |
| ------------- | ------------- |
| kNN | 230ms  |
| Approximate NN | 180ms  |

#### Тестовые метрики

| Precission  | Recall | F-мера |
| ------------- | ------------- | ------------- |
| 0.98 | 0.18 | 0.31 |

#### Example
"Картинка"