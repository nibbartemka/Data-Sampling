import logging


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(levelname)s:[%(asctime)s]/%(name)s/ %(funcName)s:%(lineno)d - %(message)s",
    )
