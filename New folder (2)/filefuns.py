import pandas as pd
import numpy as np

def csv2numpy(file,pre_ones=0,wanted_labels=[]):

    dataframe=pd.read_csv(file)

    if(len(wanted_labels)>0):
        idx = wanted_labels
    else:
        idx = list(dataframe.columns)
        idx.remove('Id')

    try:
        array = np.asanyarray(dataframe[idx])
    except:

        print('hi: Labels not found. Could not create array from .csv :(')

        print('TEST123: Labels not found. Could not create array from .csv :(')


        print('ERROR: Labels not found. Could not create array from .csv :(')

        return 0

    if(pre_ones==1):
        affine_array = np.ones((array.shape[0],array.shape[1]+1))
        affine_array[:,1:(array.shape[1]+1)] = array

        array = affine_array

    return array

def numpy2csv(array,outfile,testfile='None'):

    output = array.reshape(-1)

    if(testfile!='None'):

        testid = csv2numpy(testfile,wanted_labels=['Id'])
        output = np.column_stack((testid,output))

        output_dataframe = pd.DataFrame({'Id':testid, 'y':output})

        output_dataframe.to_csv(outfile,index=False,float_format='%.30f')

    else:
        output_dataframe = pd.DataFrame(output)

        output_dataframe.to_csv(outfile,index=False,float_format='%.30f',header=False)

    return output


