import pandas as pd

import ast
from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def run(test_path):
    data_path = "vprod_train"
    df_TRAIN_SAL = pd.read_csv(f"{data_path}/TRAIN_SAL.csv")

    SAMPLES = 50_000  # для ускорения обучения используется только 50_000 строк

    # Удаляем ненужные столбцы с минимальной и максимальной зарплатой
    df_TRAIN_SAL = df_TRAIN_SAL.drop(columns=["salary_min", "salary_max"])

    # Разделяем данные на обучающую и валидационную выборки
    train, val = train_test_split(
        df_TRAIN_SAL.iloc[:SAMPLES], test_size=0.2, random_state=228
    )

    # Выделяем целевую переменную (зарплату)
    y_train = train["salary"]
    y_val = val["salary"]

    # Выделяем признаки (все столбцы, кроме зарплаты)
    X_train = train.drop(columns=["salary"])
    X_val = val.drop(columns=["salary"])

    # Список текстовых признаков, которые будут использоваться в модели CatBoost
    text_features = [
        "additional_requirements",
        "position_requirements",
        "position_responsibilities",
        "full_company_name",
        "education_speciality",
        "other_vacancy_benefit",
        "required_certificates",
    ]

    import re


    def add_features(X_copy):
        """
        Функция добавляет новые признаки в датафрейм X_copy.

        Args:
            X_copy: Исходный датафрейм.

        Returns:
            Датафрейм с добавленными признаками.
        """
        X = X_copy.copy()

        rich_cities = {
            "Москва": (55.44, 37.36),
            "Санкт-Петербург": (59.57, 30.19),
            "Новосибирск": (55.04, 82.93),
            "Екатеринбург": (56.85, 60.62),
            "Тюмень": (57.09, 65.32),
            "Краснодар": (45.02, 38.59),
            "Сургут": (61.41, 73.26),
            "Красноярск": (56.00, 92.52),
            "Кемерово": (55.33, 86.08),
            "Иркутск": (52.17, 104.18),
        }

        for name, (x, y) in rich_cities.items():
            X[f"Расстояние_до_{name}"] = np.sqrt(
                np.square(X["vacancy_address_latitude"] - x)
                + np.square(X["vacancy_address_longitude"] - y)
            )

        """
        Если measure_type == Nan -> RUBLE
        После этого переводим премию в рубли за месяц (соответственно Annual делим на 12, Quater на 4, Monthly не трогаем
        Создаем признак premium_type - ABSENCE/PERCENT/RUBLES
        Создаем признак premium - величина премии в месяц
        """
        X["measure_type"] = X["measure_type"].fillna(value="RUBLE")
        X["has_premium"] = ~(
            (X["additional_premium"] == 0)
            | (X["additional_premium"].isna())
            | (X["bonus_type"].isna())
        )
        X["premium"] = 0
        X["premium_type"] = X["bonus_type"]
        for i, row in tqdm(X.iterrows()):
            if not row["has_premium"]:
                X.at[i, "premium_type"] = "ABSENCE"
                continue
            X.at[i, "premium"] = row["additional_premium"]

            if row["bonus_type"] == "ANNUAL":
                X.at[i, "premium"] /= 12
            elif row["bonus_type"] == "QUARTERLY":
                X.at[i, "premium"] /= 3

        X = X.drop(
            columns=["has_premium", "bonus_type", "measure_type", "additional_premium"]
        )

        X["company"] = (
            X["company"]
            .replace("false", "False", regex=True)
            .replace("true", "True", regex=True)
            .apply(ast.literal_eval)
        )

        def f(x):
            if "email" not in x.keys():
                return "nan"
            if "@" not in x["email"]:
                return "nan"
            return x["email"].split("@")[1].split(".")[0]

        X["domain"] = X["company"].apply(f)

        X["has_hr_agency"] = X["company"].apply(itemgetter("hr-agency"))

        def g(x):
            if type(x) != str:
                return False
            return x.lower() == "отдел кадров"

        X["contact_person"] = X["contact_person"].apply(g)

        # X = X.drop(columns=['bonus_type', 'measure_type', 'additional_premium'])

        X = X.drop(columns=["busy_type"])  # busy_type == schedule_type

        X["company_inn"] = X["company_inn"].astype(str)

        # юзлесс по MI скору / feature importance
        X = X.drop(
            columns=[
                "contactList",
                "deleted",
                "foreign_workers_capability",
                "is_moderated",
                "oknpo_code",
                "regionNameTerm",
                "company_name",
                "is_uzbekistan_recruitment",
                "retraining_condition",
                "publication_period",
                "retraining_grant_value",
                "career_perspective",
                "visibility",
            ]
        )

        X = X.drop(columns=["company"])  # это dict

        X = X.drop(columns=["id"])

        for text_feature in text_features:
            X[text_feature] = (
                X[text_feature]
                .replace("</p>", "", regex=True)
                .replace("<p>", "", regex=True)
                .replace("<br/>", "", regex=True)
                .replace("</li>", "", regex=True)
                .replace("<li>", "", regex=True)
            )

        X = X.drop(columns=["change_time"])

        """
        Размечаем тип компании по ключевым словам из названия
        Информация о сокращениях из самого датасета и, например, отсюда: https://www.prlib.ru/spisok-sokrashcheniy-v-naimenovaniyah-uchrezhdeniy
        """
        X["company_type"] = "Other"
        types = {
            "State": [
                "государ",
                "муницип",
                "федерал",
                "областн",
                "республикан",
                "национальн",
                "казенн",
                "министерство",
                "бюджетн",
                "бу ",
                "буз ",
                "гбу ",
                "гаоу ",
                "гапоу ",
                "гаук ",
                "гбоу ",
                "мбоу ",
                "маоу ",
                "мбу ",
                "фгу ",
                "фгоу ",
                "фгуп ",
                "гу ",
                "гуз ",
                "мбдоу ",
                "уфпс",
            ],
            "Individual": ["ип", "индивидуаль", "предпринимат", "физическ"],
            "OOO": ["общество с ограниченной ответственностью", "ооо"],
            "OAO": ["открытое акционерное общество", "оао"],
            "AO": ["акционерное общество", "ао"],
        }

        for i, row in tqdm(X.iterrows()):
            if isinstance(row["full_company_name"], float):
                continue
            name = row["full_company_name"].lower()
            for type_name, keywords in types.items():
                ok = False
                for keyword in keywords:
                    if keyword in name:
                        X.at[i, "company_type"] = type_name
                        ok = True
                        break
                if ok:
                    break

        X["num_languages"] = X["languageKnowledge"].apply(
            lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0
        )
        X["has_english"] = X["languageKnowledge"].apply(
            lambda x: "английский" in str(x).lower() if isinstance(x, str) else False
        )

        X["num_hard_skills"] = X["hardSkills"].apply(
            lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0
        )
        X["num_soft_skills"] = X["softSkills"].apply(
            lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0
        )

        X["education_level"] = X["education"].map(
            {
                "Высшее": 3,
                "Среднее профессиональное": 2,
                "Среднее": 1,
                "Среднее общее": 0.8,
                "Неполное высшее": 2.5,
                "Основное общее": 0.5,
                "Незаконченное высшее": 2.2,
                "Не указано": 1.1,
            }
        )

        def extract_years(exp):
            if pd.isna(exp):
                return 0
            match = re.search(r"(\d+)", str(exp))
            return int(match.group(1)) if match else 0

        X["required_years"] = X["required_experience"].apply(extract_years)

        X["has_accommodation"] = X["accommodation_capability"].notna()
        X["accommodation_score"] = (
            X["accommodation_type"]
            .map({"DORMITORY": 1, "FLAT": 2, "ROOM": 3, "HOUSE": 4})
            .fillna(0)
        )

        X["is_flexible"] = X["schedule_type"].apply(lambda x: "гибкий" in str(x).lower())
        X["is_shift"] = X["schedule_type"].apply(lambda x: "сменный" in str(x).lower())

        X["has_transport_compensation"] = X["transport_compensation"].notna()

        X["req_length"] = X["position_requirements"].str.len()
        X["resp_length"] = X["position_responsibilities"].str.len()

        return X

    X_train = add_features(X_train)
    X_val = add_features(X_val)

    cat_features = X_train.columns[X_train.dtypes == object]

    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna("nan")
            X_val[col] = X_val[col].fillna("nan")

    model = CatBoostRegressor(
        iterations=1_000,
        learning_rate=1e-1 * 3,
        depth=4,
        min_data_in_leaf=300,
        eval_metric="RMSE",
        # eval_metric=CustomMetric(),
        text_features=list((set(text_features) & set(list(X_train.columns)))),
        cat_features=list((set(cat_features) & set(list(X_train.columns)))),
        # task_type="GPU",
        # devices="0:1",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        # X_full, y_full,
        verbose=100,
    )


    # Загрузка данных
    df_TEST_SAL = pd.read_csv(f"{test_path}/TEST_SAL.csv")

    df_TEST_SAL['full_company_name'].loc[df_TEST_SAL['full_company_name'].isna() | df_TEST_SAL['full_company_name'] == ""] = df_TEST_SAL['company_inn']

    df_TEST_SAL = df_TEST_SAL

    X_test = df_TEST_SAL.copy()
    X_test = add_features(X_test)
    cat_features = X_train.columns[X_train.dtypes == object]
    for col in cat_features:
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna("nan")

    predictions = model.predict(X_test)

    df_TEST_RES = pd.read_csv(f"{test_path}/TEST_RES.csv")

    def create_submission_part(test_df, name_of_predict_column, values):
        submission = pd.DataFrame([])
        submission['id'] = test_df['id']
        submission[name_of_predict_column] = values
        return submission

    predict_res_part = pd.read_csv("./predict.csv")['answer']

    submission_SAL_part = create_submission_part(df_TEST_SAL, name_of_predict_column="salary", values=predictions)
    submission_RES_part = create_submission_part(df_TEST_RES, name_of_predict_column="job_title", values=predict_res_part)
    # submission_RES_part = create_submission_part(df_TEST_RES, name_of_predict_column="job_title", values=['абоба'] * len(predict_res_part))

    def create_submission(RES_part, SAL_part):
        RES_part['task_type'] = 'RES'
        SAL_part['task_type'] = 'SAL'
        submission = pd.concat([RES_part, SAL_part], axis=0)
        return submission


    submission = create_submission(submission_RES_part, submission_SAL_part)

    submission.to_csv('submission.csv', index=False)

