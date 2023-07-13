import logging

import fire

from hlm12erc import serving

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(serving.CLI)
