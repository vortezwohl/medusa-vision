import shutil

import gdown

from medusa_resources import *
from medusa_resources.util import *
from medusa_resources.storage import *
from medusa_resources.util.logger import log
from medusa_resources.exception import RollbackError


def install_resources(rollback_retries: int = 3) -> bool:
    try:
        for weight in WEIGHTS:
            weight_url = f'{MEDUSA_RESOURCES_URL}/{weight}'
            weight_file = f'{MEDUSA_STORAGE_WEIGHTS}/{weight}'
            if not file_exists(weight_file):
                gdown.download(weight_url, weight_file, quiet=False)
        for opencv in OPENCV:
            opencv_url = f'{MEDUSA_RESOURCES_URL}/{opencv}'
            opencv_file = f'{MEDUSA_STORAGE_OPENCV}/{opencv}'
            if not file_exists(opencv_file):
                gdown.download(opencv_url, opencv_file, quiet=False)
        return True
    except:
        log.error('Download failed. Clearing cached resources...')
        count = 0
        while not uninstall_resources():
            count += 1
            if count >= rollback_retries:
                raise RollbackError(f'Cached resources remains. Please manually remove '
                                    f'{MEDUSA_STORAGE_OPENCV} and {MEDUSA_STORAGE_WEIGHTS}, '
                                    f'or use medusa-clear instead.')
        return False


def uninstall_resources() -> bool:
    try:
        if file_exists(MEDUSA_STORAGE_OPENCV):
            shutil.rmtree(MEDUSA_STORAGE_OPENCV)
        if file_exists(MEDUSA_STORAGE_WEIGHTS):
            shutil.rmtree(MEDUSA_STORAGE_WEIGHTS)
    except PermissionError:
        raise RollbackError(f'Cached resources remains. Please manually remove '
                            f'{MEDUSA_STORAGE_OPENCV} and {MEDUSA_STORAGE_WEIGHTS}, '
                            f'or use medusa-clear instead.')
    res = not file_exists(MEDUSA_STORAGE_OPENCV) and not file_exists(MEDUSA_STORAGE_WEIGHTS)
    if res:
        log.info('Cached resources cleared.')
    return res
