#!/usr/bin/env python3

import logging

########################################
# Logging
########################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)
