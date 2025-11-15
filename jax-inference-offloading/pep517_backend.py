from setuptools import build_meta as _orig


def _build_protos() -> None:
    from grpc.tools import command
    command.build_package_protos(".", strict_mode=True)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _build_protos()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _build_protos()
    return _orig.build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _build_protos()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _build_protos()
    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _build_protos()
    return _orig.prepare_metadata_for_build_editable(metadata_directory, config_settings)


