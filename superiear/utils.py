import os
import inspect
import functools

def ensure_dir_exists(func):
  """
  Ensures that all args corresponding to directory paths
  (= ending on _dir) exist and otherwise creates them.

  Currently does not support kwargs.
  """
  @functools.wraps(func)
  def _ensure_dir_exists_wrapper(*args, **kwargs):
    for i, arg in enumerate(inspect.getfullargspec(func)[0]):
      if arg.endswith("_dir"):
        if not os.path.exists(args[i]):
          os.mkdir(args[i])
    return func(*args, **kwargs)
  return _ensure_dir_exists_wrapper