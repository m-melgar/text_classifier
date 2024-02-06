import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import TypedDict, Optional

from setfit import SetFitTrainer

from custom_datasets.base_dataset import get_train_n_test_Elena
from model.model import hp_space, model_init


class HyperparameterSpace(TypedDict):
    learning_rate: tuple[float, float]
    num_epochs: tuple[int, int]
    batch_size: list
    num_iterations: list
    seed: tuple[int, int]
    max_iter: tuple[int, int]
    model_id: list[str]


class ExperimentSettings(TypedDict):
    exp_name: Path
    annotations_train: list
    annotations_test: list
    out_path: Path
    load_weights: Optional[Path]
    batch_size: int
    lr: float
    epochs: int
    early_stopping: int
    wandb_project: str
    wand_entity: str
    save_every: int
    hyperparameter_space: HyperparameterSpace


def setup() -> ExperimentSettings:
    """
    Sets up argument parser and experiment settings.
    Basically, it returns a dictionary with all the experiment settings.

    :return: ExperimentSettings(TypedDict)
    """
    parser = ArgumentParser(description="Few shot text classifier.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file", type=str, help="Path to the configuration file for the experiment", )
    args = parser.parse_args()

    with open(args.config_file, 'r') as f_params:
        exp: ExperimentSettings = json.load(f_params)

    # Convert strings to path type

    exp["out_path"] = Path(exp["out_path"])
    exp["load_weights"] = Path(exp["load_weights"])

    exp["exp_name"] = Path(exp["exp_name"])

    return exp


def main(exp: ExperimentSettings) -> None:
    weight_path = exp["load_weights"]

    # train_dataset, test_dataset = get_train_n_test()
    train_dataset, test_dataset = get_train_n_test_Elena(config=exp, clean_dataset=False)

    trainer = SetFitTrainer(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model_init=model_init,
    )

    best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=25)
    print(best_run.hyperparameters)

    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    trainer.train()

    metrics = trainer.evaluate()
    print("*" * 5, "*" * 5, "*" * len("BEST MODEL"))
    print("*" * 5, "BEST MODEL", "*" * 5)
    print(f"model used: {best_run.hyperparameters}")
    print(f"train dataset: {len(train_dataset)} samples")
    print(f"accuracy: {metrics['accuracy']}")

    # save model
    best_model = trainer.model
    best_model.save_pretrained(str(weight_path))
    print(f"DONE TRAINING!\nModel saved at {weight_path}")


# model used: sentence-transformers/all-mpnet-base-v2
# train dataset: 32 samples
# accuracy: 0.873421052631579


if __name__ == '__main__':
    main(setup())
