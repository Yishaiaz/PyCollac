import numpy as np
import pandas as pd


class vectorsPreProcces:


    def same_length_vectors(self, v1:pd.DataFrame, v2:pd.DataFrame):
        if len(v1) > len(v2):
            new_array = np.zeros(shape=v1.shape)
            new_array[:len(v2)] = v2.copy()

            return v1, pd.DataFrame(new_array, columns=v1.columns)

        elif len(v2) > len(v1):
            new_array = np.zeros(shape=v2.shape)
            new_array[:len(v1)] = v1.copy()

            return v2, pd.DataFrame(new_array, columns=v2.columns)

        else:
            return v1, v2