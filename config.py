import os
from configparser import ConfigParser

root_dir = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(root_dir, "private.ini")
cfg = ConfigParser()

if os.path.exists(config_file):
    cfg.read(config_file)
else:
    cfg = None

if cfg:
    if cfg.has_section('openai'):
        openai = dict(cfg.items('openai'))
        OPENAI_API_KEY = openai.get('openai_api_key', '')
    else:
        OPENAI_API_KEY = ''

else:
    OPENAI_API_KEY = ''

MAX_RETRIES = 5
RETRY_AFTER = 2
