import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="test_without_ckpt.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.testing_without_ckpt_pipeline import test_without_ckpt

    # Applies optional utilities
    utils.extras(config)

    # Evaluate model
    return test_without_ckpt(config)


if __name__ == "__main__":
    main()
