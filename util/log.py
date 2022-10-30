import os
import logging
import logging.handlers
import datetime

# init log dir
LOG_DIR = "./log"
folder = os.path.exists(LOG_DIR)
if not folder:
    os.makedirs(LOG_DIR)     



logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)
log_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] @%(filename)s:%(lineno)d: %(message)s"
)


rf_handler = logging.handlers.TimedRotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"),
    when="midnight", interval=1, backupCount=7,
    atTime=datetime.time(0,0,0,0)
)
rf_handler.setFormatter(log_fmt)

cf_handler = logging.StreamHandler()
cf_handler.setFormatter(log_fmt)

logger.addHandler(rf_handler)
logger.addHandler(cf_handler)