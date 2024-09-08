# ML Reference - предсказание зарплатных ожиданий по резюме

## Запуск решения
Основным файлом является `main.py`. Пример запуска решения с обучением и предсказанием:
```bash
python main.py --data path_to_vprod_train.zip --model model.pt --test-path .
```

```
usage: main.py [-h] --data DATA --model MODEL [--fetch-pretrained] [--result RESULT] --test-path TEST_PATH

train a model

options:
  -h, --help             show this help message and exit
  --data DATA            path to train.zip
  --model MODEL          path to model.pt
  --fetch-pretrained     whether to fetch pretrained model weights
  --test-path TEST_PATH  path to directory with TEST_RES.csv
```

## Разбор JOB_LIST.csv
См. файл `jobs_clustering.ipynb`
