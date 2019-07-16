import numpy as np
import pandas as pd


'''
Create time series data pipeline
'''

def TimeSeriesData(data, label, slide):
    '''
    Create time series sliding data for model input
    data: 2D array, [batch size, features], may contain past label as feature
    label: 1D array, [batch size]. data & label should at same time line in each row
    slide: integer, slide window for data. Use 1~n past days to predict (n + 1) data value
    output: time pipeline dataset, size [(len(data) - slide + 1), features * slide]
    '''
    
    pipeline = [] # to store the ouptut time series data pipeline 
    out_size = data.shape[0] - slide  # count of embedded time series to pipeline
    #out_size = 5
    features = data.shape[1]
    for i in range(out_size):
        if features == 1:
            embedd = [data[i]]
            for j in range(slide - 1):
                 embedd = embedd + [data[i + (j + 1)]]
        elif features > 1:
            embedd = list(data[i])
            for j in range(slide - 1):
            #embedd = np.concatenate((embedd, data[i + (j + 1)]), axis = 0
                 embedd = embedd + list(data[i + (j + 1)])
        
        pipeline.append(embedd)
    
    pipeline = np.array(pipeline).reshape(-1, features * slide)
    label = label[slide:]
    
    return pipeline, label







