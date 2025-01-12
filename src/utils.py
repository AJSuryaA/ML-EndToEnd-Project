import os
import sys
import dill
from exception import CustomException

import pandas as pd, numpy as np

def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_path)

    except Exception as e:
        return CustomException(e, sys)