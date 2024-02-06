import json
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import TypedDict, Optional

from setfit import SetFitModel


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
    annotations_train: Path
    annotations_test: Path
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
    
    start = time.time()
    model = SetFitModel.from_pretrained(str(weight_path))
    print(f"Loaded model from {str(weight_path)}")

    preds = model(["Today the Prime Minister reported  the invasion of Australia.",
                   "The new local team lost the final competition",
                   "You would never imagined that the inflation was caused by angry rabbits",
                   "New product was developed emitting zero waste"])
    print("PREDICTIONS\n", preds)
    end = time.time()
    print(f"Processing time {end - start} s")


if __name__ == '__main__':
    main(setup())
