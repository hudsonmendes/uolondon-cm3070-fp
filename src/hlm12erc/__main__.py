# Python Built-in Modules
import logging

# Third-Party Libraries
import fire
from tqdm import tqdm

# My Packages and Modules
from hlm12erc import serving

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tqdm.pandas()
    fire.Fire(serving.CLI)
