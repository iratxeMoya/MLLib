import logging
from colorama import Fore, Style

DFORMAT = '[%(filename)s -> %(funcName)s():%(lineno)s] %(levelname)s - %(message)s'
FORMAT = '[%(levelname)s]\t%(message)s'

class CustomFormatter(logging.Formatter):
    debFormat = DFORMAT
    format = FORMAT
    FORMATS = {
        logging.DEBUG: Fore.WHITE + debFormat + Fore.RESET,
        logging.INFO: Fore.WHITE + format + Fore.RESET,
        logging.WARNING: Fore.YELLOW + format + Fore.RESET,
        logging.ERROR: Fore.RED + format + Fore.RESET,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + format + Fore.RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def getMyLogger():
    logger = logging.getLogger('MLLib')
    logger.setLevel('INFO')

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel('INFO')
        ch.setFormatter(CustomFormatter())
        
        logger.addHandler(ch)
    
    logger.propagate = False
    
    return logger

logger = getMyLogger() 