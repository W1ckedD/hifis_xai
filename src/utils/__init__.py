import yaml
from colorama import Fore, Style
from datetime import datetime

def load_config(path: str = 'config.yml') -> dict:
  """
  Load the configuration file.

  Args:
    path (str): The path to the configuration file.

  Returns:
    dict: The configuration dictionary.
  """
  with open(path, 'r') as file:
    cfg = yaml.safe_load(file)
  return cfg

def log_time(msg: str, color: str = 'm') -> None:
  if color == 'm':
    _color = Fore.MAGENTA
  elif color == 'g':
    _color = Fore.GREEN
  elif color == 'r':
    _color = Fore.RED
  elif color == 'b':
    _color = Fore.BLUE
  elif color == 'y':
    _color = Fore.YELLOW
  elif color == 'c':
    _color = Fore.CYAN
  else:
    _color = Fore.RESET
  print(f"{_color}{msg}{Style.RESET_ALL}: {datetime.utcnow().strftime('%H:%M:%S')}")