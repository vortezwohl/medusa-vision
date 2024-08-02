import os

ROOT = os.path.dirname(__file__)
if str(ROOT).__contains__('site-packages'):
    CONFIG = f'{ROOT}\\..\\..\\..\\..\\medusa-vision-static'
else:
    CONFIG = f'{ROOT}\\..\\..\\medusa-vision-static'
