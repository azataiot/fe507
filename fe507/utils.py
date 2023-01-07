# fe507 / utils.py
# Created by azat at 5.01.2023
# package level logging
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
