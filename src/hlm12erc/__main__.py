import logging

import fire
from tqdm import tqdm

from hlm12erc import serving

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tqdm.pandas()
    fire.Fire(serving.CLI)
