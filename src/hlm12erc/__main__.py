# Python Built-in Modules
import logging
import warnings

# Third-Party Libraries
import fire
import transformers
from tqdm import tqdm

# My Packages and Modules
from hlm12erc import serving

if __name__ == "__main__":
    # logging settings
    logging.basicConfig(level=logging.INFO)
    transformers.logging.set_verbosity(transformers.logging.ERROR)
    logging.getLogger(transformers.Trainer.__module__).setLevel(logging.INFO)
    warnings.simplefilter("ignore")

    # pandas/tqdm
    tqdm.pandas()

    # launching CLI
    fire.Fire(serving.CLI)
