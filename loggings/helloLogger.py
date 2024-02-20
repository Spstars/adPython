import logging

logger = logging.getLogger(__name__)
#baseConfig를 사용하지 않게 된다.
# logger.propagate= False
logger.info('this is logger from helloLogger.py')

stream_h = logging.StreamHandler()
file_h = logging.FileHandler("file.log")

stream_h.setLevel(logging.WARNING)
file_h.setLevel(logging.ERROR)

stream_f = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
stream_h.setFormatter(stream_f)
file_h.setFormatter(stream_f)

logger.addHandler(stream_h)

logger.addHandler(file_h)

logger.warning("this is a warning")
logger.error("this is an error")