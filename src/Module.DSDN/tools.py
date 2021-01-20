import os
import shutil


def create_dir(dir: str):
    """
    Create dir if not exist
    Args:
        dir: path to the directory
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def remove_dir(dir: str):
    """
    Remove directory
    Args:
        dir: path to the directory
    """
    shutil.rmtree(dir)


def del_file(file: str):
    """
    Remove file
    Args:
        file: path to the file
    """
    os.remove(file)


def status_message(i: int, n: int, mess: str):
    """
    Prints status message
    Args:
        i: current progress
        n: total number
        mess: message to print
    """
    print("{}/{} - {}".format(i, n, mess))


def refactor(file: str, extension: str) -> str:
    """
    Change extension of the file
    Args:
        file: path to the file
        extension: name of the extension to change

    Returns:
        path to file with changed extension
    """
    ext_len = len(file.split('.')[-1])
    return file[:-(ext_len + 1)] + ".{}".format(extension)

