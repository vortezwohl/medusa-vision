__version__ = '0.6.3'
__author__ = 'vortezwohl'
__email__ = 'vortezwohl@proton.me'

from medusa_resources.storage.storage_manager import install_resources
from medusa_resources.exception import ResourceDownloadError
from medusa_resources.storage import MEDUSA_RESOURCES_GITHUB

if install_resources(rollback_retries=3):
    from .test import webcam_test
else:
    raise ResourceDownloadError(f'Failed Downloading resources. '
                                f'Please try again or install manually from '
                                f'{MEDUSA_RESOURCES_GITHUB}.')
