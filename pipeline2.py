import pandas as pd
import zipfile
import ast
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import requests
import shutil
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from sklearn.metrics import precision_score, recall_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np
from sklearn import metrics
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModelForSequenceClassification,
)

from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification
from transformers import (
    BertForSequenceClassification,
    AdamW,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    DistilBertTokenizer,
)

# Глобальные переменные
TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "sergeyzh/rubert-mini-sts"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-4
CLASSES_COUNT = 100
# fmt: off
SHORT_NAMES = [
    "специал", "менедж", "бухгалт", "инженер", "оператор", "продав", "водит", "админ",
    "директор", "замест", "инспект", "мастер", "кассир", "машинист", "слесар", "руковод",
    "воспитат", "учител", "эконом", "секретар", "повар", "кладовщ", "подсобн", "уборщ",
    "юрискон", "охран", "техник", "электромонт", "сторож", "заведующ", "консульт", "монтаж",
    "преподават", "лаборант", "контрол", "юрист", "механик", "представит", "диспетч",
    "делопроизвод", "эксперт", "официант", "грузчик", "управля", "электрогазосвар", "трактор",
    "педагог", "товаровед", "сотрудн", "электромонтаж", "предприним", "дизайн", "инструкт",
    "психолог", "медсестр", "сборщик", "ассистент", "помощник воспитат", "бармен",
    "начальник отдел", "аппаратч", "программ", "дворник", "уборщиц", "помощник руковод",
    "методист", "почтальон", "электромехан", "парикмах", "технолог", "швея", "горничн",
    "фельдшер", "плотник", "бригадир", "пекарь", "механизатор", "кухонный рабоч", "электрик",
    "тренер", "дорожный рабоч", "курьер", "наладч", "медсестр", "комплектовщ", "прораб",
    "энергетик", "библиотек", "социальный работн", "электромонт", "рабочий по благоустройств",
    "токарь", "стрелок", "проводник", "документовед", "электрослесар", "контролер", "сварщик",
    "стропальщ", "регистрат",
]
# fmt: on

print(f"PyTorch использует {TORCH_DEVICE}")


def download_file(url, dest):
    """
    Скачивает файл по заданному URL.

    Параметры:
    ----------
    url : str
        URL для скачивания
    dest : str
        Путь до файла для сохранения
    """
    print(f"Скачиваем {dest}...")
    with requests.get(url, stream=True) as r:
        with open(dest, "wb") as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
    return dest


class TrainDataset(Dataset):
    """
    Класс для создания обучающего датасета для модели BERT.

    Атрибуты:
    ----------
    dataframe : pd.DataFrame
        Датафрейм с данными для обучения.
    tokenizer : transformers.PreTrainedTokenizer
        Токенизатор для преобразования текста в токены.
    max_len : int
        Максимальная длина последовательности токенов.
    """

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = self.data["text"]
        self.achievements = self.data["achievements"]
        self.targets = self.data["label"]
        self.index = np.array(self.data.index)
        self.max_len = max_len

    def __len__(self):
        """Возвращает количество образцов в датасете."""
        return len(self.text)

    def get(self, text, prefix=""):
        """
        Преобразует текст в токены и возвращает словарь с тензорами.

        Параметры:
        ----------
        text : str
            Текст для токенизации.
        prefix : str, optional
            Префикс для названий ключей в возвращаемом словаре (по умолчанию "").

        Возвращает:
        ----------
        dict
            Словарь с тензорами input_ids, attention_mask и token_type_ids.
        """
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            prefix + "ids": torch.tensor(ids, dtype=torch.long),
            prefix + "mask": torch.tensor(mask, dtype=torch.long),
            prefix + "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

    def __getitem__(self, index):
        """
        Возвращает один образец данных по индексу.

        Параметры:
        ----------
        index : int
            Индекс образца.

        Возвращает:
        ----------
        dict
            Словарь с токенизированным текстом, достижениями и целевыми метками.
        """
        text = str(self.text[index])
        achievements = str(self.achievements[index])

        return (
            self.get(text)
            | self.get(achievements, prefix="a")
            | {
                "targets": torch.tensor(self.targets[index], dtype=torch.float),
                "index": index,
            }
        )


class BERTClass(torch.nn.Module):
    """
    Класс модели на основе BERT для классификации текстов.

    Атрибуты:
    ----------
    num_classes : int
        Количество классов для классификации.
    """

    def __init__(self, num_classes):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained(MODEL_NAME)
        self.la = BertModel.from_pretrained(MODEL_NAME)
        self.l2 = torch.nn.Linear(312 * 2, num_classes)

    def forward(self, ids, mask, token_type_ids, aids, amask, atoken_type_ids):
        """
        Прямой проход (forward) через модель.

        Параметры:
        ----------
        ids : torch.Tensor
            Тензор с input_ids для основного текста.
        mask : torch.Tensor
            Тензор с attention_mask для основного текста.
        token_type_ids : torch.Tensor
            Тензор с token_type_ids для основного текста.
        aids : torch.Tensor
            Тензор с input_ids для текста достижений.
        amask : torch.Tensor
            Тензор с attention_mask для текста достижений.
        atoken_type_ids : torch.Tensor
            Тензор с token_type_ids для текста достижений.

        Возвращает:
        ----------
        torch.Tensor
            Тензор с предсказанными логитами для каждого класса.
        """
        x = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            "pooler_output"
        ]
        xa = self.la(aids, attention_mask=amask, token_type_ids=atoken_type_ids)[
            "pooler_output"
        ]

        # Объединяем выходы двух моделей BERT и пропускаем через линейный слой
        x = self.l2(torch.cat([x, xa], dim=1))
        x[:, -1] = (
            -10000
        )  # Принудительно задаем очень низкое значение для последнего класса
        return x


def calculate_accuracy(other_idx, preds, targets, num_classes):
    """
    Вычисляет точность (F1-score) для многоклассовой классификации.

    Параметры:
    ----------
    other_idx : int
        Индекс класса "другие".
    preds : torch.Tensor
        Предсказанные классы.
    targets : torch.Tensor
        Истинные классы.
    num_classes : int
        Общее количество классов.

    Возвращает:
    ----------
    float
        Значение F1-score.
    """
    preds[preds == other_idx] = other_idx + 1000  # Исключаем класс "другие" из расчета
    return multiclass_f1_score(preds, targets, num_classes=num_classes, average="micro")


def train(
    model, training_loader, optimizer, loss_function, other_idx, num_classes, epochs
):
    """
    Обучает модель на тренировочном наборе данных.

    Параметры:
    ----------
    model : torch.nn.Module
        Модель для обучения.
    training_loader : DataLoader
        DataLoader для тренировочного набора данных.
    optimizer : torch.optim.Optimizer
        Оптимизатор для обновления весов модели.
    loss_function : callable
        Функция потерь.
    other_idx : int
        Индекс класса "другие".
    num_classes : int
        Общее количество классов.
    epochs : int
        Количество эпох для обучения.
    """
    print("Обучаем модель...")
    for epoch in range(epochs):
        past_lr = LEARNING_RATE

        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()

        accuracies = []
        for it, data in tqdm(enumerate(training_loader, 0)):
            # Изменение скорости обучения в зависимости от итерации
            if it < 4000:
                lr = 1e-4
            elif it < 10_000:
                lr = 3e-5
            elif it < 15_000:
                lr = 8e-6
            else:
                lr = 1e-6

            if it == 8_000:
                break

            if lr != past_lr:
                past_lr = lr
                for g in optimizer.param_groups:
                    g["lr"] = lr

            # Перемещение данных на устройство (GPU или CPU)
            ids = data["ids"].to(TORCH_DEVICE, dtype=torch.long)
            mask = data["mask"].to(TORCH_DEVICE, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(TORCH_DEVICE, dtype=torch.long)
            aids = data["aids"].to(TORCH_DEVICE, dtype=torch.long)
            amask = data["amask"].to(TORCH_DEVICE, dtype=torch.long)
            atoken_type_ids = data["atoken_type_ids"].to(TORCH_DEVICE, dtype=torch.long)
            targets = data["targets"].to(TORCH_DEVICE, dtype=torch.long)

            # Прямой проход через модель
            outputs = model(ids, mask, token_type_ids, aids, amask, atoken_type_ids)

            # Вычисление потерь
            loss = loss_function(outputs, targets)

            # Суммирование потерь для текущей эпохи
            tr_loss += loss.detach().cpu().item()

            # Определение предсказанных классов
            big_val, big_idx = torch.max(outputs.data, dim=1)

            # Вычисление точности (F1-score) для текущей итерации
            accuracy = (
                calculate_accuracy(other_idx, big_idx, targets, num_classes)
                .cpu()
                .item()
            )
            accuracies.append(accuracy)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # Логирование потерь и точности каждые K итераций
            K = 300
            if it % K == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = np.mean(accuracies[-1000:])

                # XXX: Здесь мы смотрим на F1 на train датасете, но
                #      так как на данный момент мы обучаемся за одну
                #      эпоху, эти данные модель видит впервые.
                print(f"Training Loss {it} steps: {loss_step}")
                print(f"Training F1 {it} steps: {accu_step}")

            # Обратное распространение ошибки и обновление весов
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def unzip_train_archive(data_zip_path):
    """
    Распаковывает архив с данными для обучения.

    Параметры:
    ----------
    data_zip_path : str
        Путь к zip-архиву с данными.
    """
    print(f"Распаковка {data_zip_path}...")
    with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
        zip_ref.extractall(".")


def predict(model, loader, classes, other_idx, decode=False):
    """
    Запускает модель на тестовом наборе данных.

    Параметры:
    ----------
    model : torch.nn.Module
        Модель для предсказания.
    loader : DataLoader
        DataLoader для тестового набора данных.
    """
    predicted = []
    indices = []

    model.eval()

    for _, data in tqdm(enumerate(loader, 0)):
        ids = data["ids"].to(TORCH_DEVICE, dtype=torch.long)
        mask = data["mask"].to(TORCH_DEVICE, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(TORCH_DEVICE, dtype=torch.long)
        aids = data["aids"].to(TORCH_DEVICE, dtype=torch.long)
        amask = data["amask"].to(TORCH_DEVICE, dtype=torch.long)
        atoken_type_ids = data["atoken_type_ids"].to(TORCH_DEVICE, dtype=torch.long)

        indices += list(data['index'])

        with torch.no_grad():
            outputs = model(
                ids, mask, token_type_ids, aids, amask, atoken_type_ids
            )

        big_val, big_idx = torch.max(outputs.data, dim=1)
        predicted.extend(big_idx.cpu().numpy())

    if decode:
        predicted = decode_array(classes, other_idx, predicted)
    
    return np.array(predicted, dtype=object)[indices]


def decode_array(classes, other_idx, a):
    """
    Восстанавливает названия профессий
    """
    for i, x in enumerate(a):
        if x == other_idx:
            a[i] = "other"
        else:
            a[i] = classes[x]
    return a


def run(
    data_zip_path=None,
    model_path=None,
    use_pretrained=False,
    pretrained_download_url="https://github.com/Dliwk/cp-ml-reference/releases/download/v1.0/model.pt",
    do_predict=True,
    prediction_file_path='prediction.csv',
):
    """
    Основная функция для запуска обучения модели.

    Параметры:
    ----------
    data_zip_path : str, optional
        Путь к zip-архиву с данными для обучения.
    model_path : str, optional
        Путь для сохранения обученной модели.
    use_pretrained : bool, optional
        Флаг для использования предобученной модели (по умолчанию False).
    pretrained_download_url : str, optional
        URL для скачивания предобученых весов модели.
    prediction_file_path : str, optional
        Путь до файла с результатами предсказания (по умолчанию prediction.csv).
    """
    data_path = "vprod_train"

    # Распаковка архива с данными
    if not Path(data_path).exists():
        unzip_train_archive(data_zip_path)

    # Загрузка данных
    print("Загружаем датасет в оперативную память...")
    data = pd.read_csv(f"{data_path}/TRAIN_RES_1.csv")  # всего их 5
    # test_data = pd.read_csv(f'TEST_RES.csv')
    test_data = pd.read_csv(f'test_test_res.csv')
    # test_data = pd.read_csv(f'{data_path}/TRAIN_RES_2.csv').iloc[:100]

    # Предобработка данных
    data = data.rename(columns={"id_cv": "id"})
    data["company_name"] = data["company_name"].fillna("nan")

    test_data = test_data.rename(columns={"id_cv": "id"})
    test_data["company_name"] = test_data["company_name"].fillna("nan")

    Xu = data.groupby("id").last()
    X_test = test_data.copy()

    y = Xu["job_title"]
    X = Xu.drop(columns=["job_title"])

    X["demands"] = X["demands"].fillna("нет")
    X_test["demands"] = X_test["demands"].fillna("нет")

    # Определение классов и их индексов
    classes = list(Xu["job_title"].value_counts().index[:CLASSES_COUNT])

    class_to_idx = {}
    for i, c in enumerate(classes):
        class_to_idx[c] = i

    other_idx = len(class_to_idx)
    y = y.apply(lambda x: class_to_idx[x] if x in class_to_idx else other_idx)

    weights = torch.tensor(
        list(1.0 / Xu["job_title"].value_counts().iloc[:CLASSES_COUNT]) + [0]
    ).to(TORCH_DEVICE)

    # Объединение названия компании и достижений в один текст
    Xu["achievements_modified"] = Xu["achievements_modified"].fillna("не указано")
    Xu["achievements_modified"] = (
        "достижения: " + Xu["achievements_modified"]
    )

    X_test["achievements_modified"] = X_test["achievements_modified"].fillna("не указано")
    X_test["achievements_modified"] = (
        "достижения: " + X_test["achievements_modified"]
    )

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(Xu, y, test_size=0.2)

    # Создание датафреймов для обучения и валидации
    df_train = pd.DataFrame(
        {
            "text": X_train["demands"],
            "achievements": X_train["achievements_modified"],
            "label": y_train,
        }
    ).reset_index()

    df_val = pd.DataFrame(
        {
            "text": X_val["demands"],
            "achievements": X_val["achievements_modified"],
            "label": y_val,
        }
    ).reset_index()

    df_test = pd.DataFrame(
        {
            "text": X_test["demands"],
            "achievements": X_test["achievements_modified"],
            "label": 1,
        }
    ).reset_index()

    # Инициализация токенизатора
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Создание датасетов для обучения и валидации
    training_set = TrainDataset(df_train, tokenizer, MAX_LEN)
    validation_set = TrainDataset(df_val, tokenizer, MAX_LEN)
    test_set = TrainDataset(df_test, tokenizer, MAX_LEN)

    # Параметры для DataLoader
    train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}
    val_params = {"batch_size": VALID_BATCH_SIZE, "shuffle": True, "num_workers": 0}
    test_params = {"batch_size": VALID_BATCH_SIZE, "shuffle": False, "num_workers": 0}

    # Создание DataLoader для обучения и валидации
    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)

    # Определение количества классов
    NUM_CLASSES = len(classes) + 1

    # Инициализация модели
    model = BERTClass(NUM_CLASSES).to(TORCH_DEVICE)

    # Определение функции потерь и оптимизатора
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, eps=1e-6)

    if use_pretrained:
        if not Path(model_path).exists():
            download_file(pretrained_download_url, model_path)

        model.load_state_dict(torch.load(model_path, map_location=TORCH_DEVICE))
    else:
        # Обучение модели
        train(
            model,
            training_loader,
            optimizer,
            loss_function,
            other_idx,
            NUM_CLASSES,
            EPOCHS,
        )

        # Сохранение обученной модели
        torch.save(model.state_dict(), model_path)

    if do_predict:
        result = predict(model, test_loader, classes, other_idx, decode=True)
        result_df = pd.DataFrame({'answer': result})
        result_df.to_csv(prediction_file_path)

