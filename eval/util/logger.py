import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
st_handler = logging.StreamHandler()
logger.addHandler(st_handler)