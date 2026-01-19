import os

_REQUIREMENT_PATH = os.path.join(os.path.dirname(__file__), 'envs_DLTK')

XMIPP_CONDA_ENVS = {
  "xmipp_DLTK_v0.3": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_DLTK_v0.3.yml'),
    "xmippEnviron": True,
    "versionId": 1 # Increment when the corresponding yml is modified
  },

  "xmipp_MicCleaner": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_MicCleaner.yml'),
    "xmippEnviron": False,
    "versionId": 1 # Increment when the corresponding yml is modified
  },

  "xmipp_deepEMhancer": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_deepEMhancer.yml'),
    "xmippEnviron": False,
    "versionId": 1 # Increment when the corresponding yml is modified
  },
  
  "xmipp_pyTorch": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_pyTorch.yml'),
    "xmippEnviron": True,
    "versionId": 1 # Increment when the corresponding yml is modified

  },

  "xmipp_DLTK_v1.0": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_DLTK_v1.0.yml'),
    "xmippEnviron": True,
    "versionId": 1 # Increment when the corresponding yml is modified
  },

  "xmipp_jax": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_jax.yml'),
    "xmippEnviron": True,
    "versionId": 1 # Increment when the corresponding yml is modified
  },
}
