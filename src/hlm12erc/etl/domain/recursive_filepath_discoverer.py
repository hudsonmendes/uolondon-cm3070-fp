import pathlib


class RecursiveFilepathDiscoverer:
    """
    Discover the filepath for the given filename.
    """

    root: pathlib.Path

    def __init__(self, root: pathlib.Path) -> None:
        """
        Create a new filepath discoverer looking at the given root directory.
        :param root: The root directory to discover the filepath from.
        """
        self.root = root

    def __call__(self, filename: str) -> pathlib.Path:
        """
        Discover the filepath for the given filename.
        :param filename: The filename to discover the filepath for.
        :return: The filepath for the given filename.
        """
        for path in self.root.rglob(filename):
            return path
        raise FileNotFoundError(f"Could not find {filename} in {self.root}")
