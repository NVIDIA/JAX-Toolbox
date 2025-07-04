import logging
from .container import Container
from .docker import DockerContainer
from .pyxis import PyxisContainer
from .local import LocalContainer


def make_container(
    runtime: str, url: str, mounts: list, logger: logging.Logger, **kwargs
) -> Container:
    """
    This function craetes a container objects, based on the specified runtime

    Args:
        runtime (str): The container runtime to use (e.g., 'docker', 'pyxis', 'local').
        url (str): The URL of the container.
        mounts (list): List of mounts to be used in the container.
        logger (logging.Logger): Logger instance for logging messages.
        **kwargs: Additional keyword arguments for specific container types.

    Returns:
        Container: A container class associated with the specified runtime.
    """
    if runtime == "local":
        return LocalContainer(logger=logger)

    container_impl = DockerContainer if runtime == "docker" else PyxisContainer
    return container_impl(url, logger=logger, mounts=mounts, **kwargs)
