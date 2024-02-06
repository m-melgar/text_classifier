import datasets
import pandas as pd
from datasets import load_dataset, concatenate_datasets

_LABELS = ['Negative', 'Neutral', 'Positive']


def get_train_n_test() -> tuple:
    """
    Returns train and test datasets.
    :return:
    """
    # Load the dataset
    dataset = load_dataset("ag_news")

    seed = 20
    labels = 4
    samples_per_label = 8
    sampled_datasets = []
    # find the number of samples per label
    for i in range(labels):
        sampled_datasets.append(
            dataset["train"].filter(lambda x: x["label"] == i).shuffle(seed=seed).select(range(samples_per_label)))

    # concatenate the sampled datasets
    train_dataset = concatenate_datasets(sampled_datasets)

    # create test dataset
    test_dataset = dataset["test"]

    return train_dataset, test_dataset


def drop_csv_columns(csv_file: str, columns: list = ['id', 'Comments']):
    df = pd.read_csv(csv_file)
    df = df.drop(columns, axis=1)
    df["label"] = df["label"].replace("Negative", _LABELS.index("Negative"))
    df["label"] = df["label"].replace("Neutral", _LABELS.index("Neutral"))
    df["label"] = df["label"].replace("Positive", _LABELS.index("Positive"))
    df.to_csv(csv_file, index=False)


def get_train_n_test_Elena(config, clean_dataset=True) -> tuple:
    """
    Returns train and test datasets.
    :return:
    """
    # clean dataset:
    if clean_dataset:
        for csv_file in config["annotations_train"]:
            drop_csv_columns(csv_file)
        for csv_file in config["annotations_test"]:
            drop_csv_columns(csv_file)

    dataset = datasets.load_dataset("csv", data_files={"train": config["annotations_train"],
                                                       "test": config["annotations_test"]})

    return dataset["train"], dataset["test"]
