import pandas
import numpy

def get_vector_of_minmaxing(df : pandas.DataFrame):
    result = pandas.DataFrame()
    for key, val in df.items():
        result[key] = {"min" : val.min(), "max" : val.max(), "std" : val.std(), "mean" : val.mean()}
    return result

def minmax(_df : pandas.DataFrame, minmaxvect : pandas.DataFrame):
    df = _df.copy()
    for key, val in df.items():
        df[key] = (df[key]-minmaxvect[key]["min"])/(minmaxvect[key]["max"]-minmaxvect[key]["min"])
    return df