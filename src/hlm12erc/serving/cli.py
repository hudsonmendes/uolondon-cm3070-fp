# Local Folders
from .cli_erc import ERCCommands
from .cli_etl import ETLCommands


class CLI:
    """
    The command line interface (or "CLI") for the HLM12ERC package.
    """

    def etl(self) -> "ETLCommands":
        """
        Return the CLI commands for Extract, Transform, Load (or "ETL").
        """
        return ETLCommands()

    def erc(self) -> "ERCCommands":
        """
        Return the CLI commands for Emotion Recognition in Conversation (or "ERC").
        """
        return ERCCommands()
