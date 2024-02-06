from setfit import SetFitModel


def model_init(params) -> SetFitModel:
    """
    Model initializer.
    :param params:
    :return:
    """
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    model_id = params.get("model_id", "sentence-transformers/all-mpnet-base-v2")
    model_params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained(model_id, **model_params)


def hp_space(trial) -> dict:
    """
    Hyperparameter space definition
    :param trial:
    :return:
    """

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32]),
        "num_iterations": trial.suggest_categorical("num_iterations", [5, 10, 20, 40, 80]),
        "seed": trial.suggest_int("seed", 1, 40),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs"]),
        "model_id": trial.suggest_categorical(
            "model_id",
            [
                # "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L12-v1",
            ],
        ),
    }
