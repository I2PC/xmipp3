import os
import traceback

BAD_IMPORT_MSG='''
Error, tensorflow/keras is probably not installed. Install it with:\n  ./scipion installb deepLearningToolkit
If gpu version of tensorflow desired, modify  SCIPION_DIR/config/scipion.conf
CUDA = True

and 

XMIPPP/xmipp.conf
CUDA=True
'''

#TODO. MODIFY THIS TO DEAL WITH CONDA ENVIRONMENTs USING CondaEnvManager
def checkIf_tf_keras_installed():
  '''
  It should just be employed within a script called on runCondaJob or runCondaCmd
  :return:
  '''
  try:
    import tensorflow as tf
  except ImportError as e:
    print(e)
    raise ValueError(BAD_IMPORT_MSG)


def updateEnviron(gpus=None):
  """ Create the needed environment for TensorFlow programs. """
  print("updating environ to select gpus: %s"%(gpus) )
  if gpus is None or gpus==-1:
    os.environ['CUDA_VISIBLE_DEVICES']="-1"
    return None
  else:
    if isinstance(gpus, str):
      if gpus.startswith("all"):
        return "all"
      elif gpus is not "":
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        if "," in gpus:
          return [int(elem) for elem in gpus.split(",")]
        else:
          return [int(elem) for elem in gpus]
      else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        return None
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
      if isinstance(gpus, list):
        return [int(elem) for elem in gpus]
      else:
        return [int(gpus)]

