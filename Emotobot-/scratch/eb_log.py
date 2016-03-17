import logging

# Emotobot Logging

def printAndDebug(msg):
    print msg
    logging.debug(msg)


def printAndLog(msg):
    print msg
    logging.info(msg)


def printAndWarn(msg):
    print msg
    logging.info(msg)


def printAndError(msg):
    print msg
    logging.error(msg)


def printAndCritical(msg):
    print msg
    logging.critical(msg)