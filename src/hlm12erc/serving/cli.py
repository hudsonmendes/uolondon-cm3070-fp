from .cli_arxiv import ArxivCommands
from .cli_erc import ERCCommands


class CLI:
    def arxiv(self) -> ArxivCommands:
        return ArxivCommands()

    def erc(self) -> ERCCommands:
        return ERCCommands()
