import numpy as N
import pandas as pd

import logging


def getRadarLength(TimeToEnd):
    """
    Returns a n-tuble with (n1, n2...)
    where n? is the number of measurements for radar ?
    To add a dataframe column do:
    df['RadarLength'] = df['TimeToEnd'].apply(getRadarLength)
    """

    tlist = map(float, TimeToEnd.split())
    nlist = [0,]
    previous_time = 9999999999
    current_radar = 0
    for it in tlist:
        if it > previous_time:
            current_radar += 1
            nlist.append(0)
            #assert(current_radar < 2)##it seems that there are cases with >2 radars
        nlist[current_radar] += 1
        previous_time = it
    return tuple(nlist)

def separate_listInColumn(x):
    """
    Returns a tuple where all the measurements are separeted by radar 
    Input:
        x : should be a panda Dataframe
                - First column must be the tuble of radar length
                - Following columns should be columns to separate

    """
    # determine if there is more than one time step
    if type(x.iloc[1]) == float:
        # There is a single time step, and the pandas read a float
        listrads = [x.iloc[1]]
    elif type(x.iloc[1]) == str:
        # There are multiple time steps, and the pandas read a string
        listrads = map(float,  x.iloc[1].split())

    # The list in then sliced by radar given in the first elements of x
    # x.iloc[0] is a tuple with the length of each radar measurement, i.e.
    # x.iloc[0][0] is the # of measurement with the 1st radar (x.iloc[0][1] for the 2nd radar)
    # The following line could be rewritten more clearly (but less efficient?) like this:
    # rad_measurements = x.iloc[0]
    # nrad1, nrad2 = x.iloc[0]
    # rad1, rad2 = listrads[:nrad1], listrads[nrad1:nrad1 + nrad2]
    #if len(x.iloc[0]) < 2:
    #    return [listrads,]
    #return listrads[:x.iloc[0][0]], listrads[x.iloc[0][0]:x.iloc[0][0] + x.iloc[0][1]]
    by_rads = [listrads[:x.iloc[0][0]], ]
    for idx in range(len(x.iloc[0]))[:-1]:
        by_rads.append(listrads[sum(x.iloc[0][:idx+1]):sum(x.iloc[0][:idx+2])])
    return tuple(by_rads)

def getIthRadar(x, iradar =1):
    """
    Returns a list of measurements for the ith radar
    Returns None if there are no ith radar
    Input:
        x : should be a panda Dataframe
                - First column must be the tuble of radar length
                (as produced by getRadarLength)
                - Second columns must contains the values to separate
        iradar : the ith radar to return the data from (default=1st)
    """
    if len(x.iloc[0])<iradar:
        return None## Or should it be NA?
    ## The longer but clearer way
    try:
        listrads = map(float,  x.iloc[1].split())
    except AttributeError:
        return (0, )
    rad_start_index = sum(x.iloc[0][:iradar-1])
    rad_stop_index = rad_start_index + x.iloc[0][iradar-1]
    return tuple(listrads[rad_start_index:rad_stop_index])

def getListReductions(x):
    """
    Returns mean, range (max - min) and # of values of the given list
    Input:
        x : should be a list or tuple
        (or something that can be turned into a numpy array)
    """
    xarray = N.array(x)
    return xarray.mean(), xarray.ptp(axis=0), len(xarray)

def getStringReductions(x):
    """
    Returns mean, range (max - min) and # of values of the given string
    Input:
        x : should be a space-separated value string
    """
    xarray = N.array(map(float, x.split()))
    return xarray.mean(), xarray.ptp(axis=0), len(xarray)


def get_dataframe_with_split_multiple_radars(input_df):
    """
    Separates the rows containing multiple radar time series into mutliple rows, introducing a new
    column identifying the group the new rows came from.
    Input:
        raw pandas dataframe, as read directly from the training dataset.
    Output:
        new pandas dataframe, with each row corresponding to an individual radar.
    """


    # Get a list of all the columns that will have to be split by radar.
    columns_to_split = list(input_df.columns)

    columns_to_split.remove('Id')       # not a time series!
    columns_to_split.remove('Expected') # not a time series!

    # Create an array of all the column names, in the same order as in the 
    # original data.
    right_column_order = ['unique_Id']
    for col in input_df.columns:
        right_column_order.append(col)
             
    # Append a new column that will represent how many radars were in the set from
    # which a given row came from.
    right_column_order.append('number_of_radars')

    # Loop over all rows of the input dataframe, splitting mutliple radar rows
    # as we go along.

    # note about the algorithm below:
    #   It seems pretty ugly to loop on an index (not very pythonic). 
    #   However, since I don't know ahead of time how many rows I'll have,
    #   a stackoverflow comment suggests that creating a list of dictionaries is faster.
    #   See: http://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe
    #

    # list of dictionaries, which will become our new dataframe
    list_new_dataframe_dict = []

    id_counter = -1  # unique identifier for our split radar entries

    num_lines = len(input_df)
    for index in range(num_lines):    

        if index%100 == 0:
            print 'doing row %i of %i ...'%(index,num_lines)

        # create a copy of the row, so we can manipulate
        # it without polluting the initial dataframe. 
        row = input_df.loc[index].copy()

        ID       = row['Id']
        expected = row['Expected']

        # don't want to pollute input dataframe! We're 
        # hacking the copy here.
        row['RadarCounts'] = getRadarLength(row['TimeToEnd'])
        number_of_radars   = len(row['RadarCounts'])

        # list of dictionaries spawned by this row
        list_newrows_dict = []

        # initialize all relevant dictionaries with 
        # "family" data, ie stuff that is the same for all radars in row. 
        for i in row['RadarCounts']:
            id_counter += 1
            list_newrows_dict.append({        'unique_Id':id_counter,
                                                     'Id':ID, 
                                       'number_of_radars':number_of_radars, 
                                               'Expected':expected          })

        # populate the new dictionaries just created above with every column data
        for col in columns_to_split:

            # get subrow so we can apply splitting methods            
            subrow = row[['RadarCounts',col]]

            # fill the dictionaries with the split data
            for array, dict in zip(separate_listInColumn(subrow),list_newrows_dict):
                dict[col] = N.array(array)


        # extend the main list of dictionaries with the entries from this row        
        list_new_dataframe_dict.extend( list_newrows_dict )

    # create the new dataframe from the list of dictionaries
    output_df = pd.DataFrame(list_new_dataframe_dict)[right_column_order].set_index('unique_Id')

    return output_df 


def get_clean_average(array):
    """
    Remove all error code and NaN values before taking average.
    If nothing is left, yield NaN.
    """

    error_codes = [ -99900.0, -99901.0, -99903.0, 999.0]

    # take out the NaN
    float_array = array[ N.where(N.isfinite(array)) ] 

    I = N.ones_like(float_array)
    for code in error_codes: 
        I *= float_array != code

    left_over_array = float_array[ N.where(I) ] 

    if len(left_over_array ) == 0:
        return N.NaN
    else:
        return N.average(left_over_array)

def get_clean_average_dataframe(input_df):
    """
    Computes the averages of time series, removing missing data (or error codes).
    When this is impossible, the missing value is replaced by the column average.
    Input:
        pandas dataframe from function "get_dataframe_with_split_multiple_radars"
    Output:
        new pandas dataframe, with averages replacing time series
    """
    # Get a list of all the columns that will have to be split by radar.
    columns_to_split = list(input_df.columns)

    dict_averages = {}
    for key in ['Id', 'Expected', 'number_of_radars']:
        columns_to_split.remove(key)       # not a time series!

        dict_averages[key] = input_df[key].values


    for col in columns_to_split:
        # compute the average of the time series
        avg = input_df[col].apply(get_clean_average).values

        # replace missing values by column average
        I_nan = N.where(N.isnan(avg))[0]

        if len(I_nan) > 0:
            I_finite = N.where(N.isfinite(avg))[0]
            finite_average = N.average(avg[I_finite])
            avg[I_nan] = finite_average 

            #print c, I_nan.shape, finite_average 
        dict_averages['avg_%s'%col] = avg


    # create the new dataframe from the list of dictionaries
    output_df = pd.DataFrame(dict_averages)

    return output_df 


    
if __name__ == "__main__":
    #print getRadarLength([5,4,3,2,1])
    #print getRadarLength([5,4,3,7,1])

    ##The following lines show how to create a new column with the length of each radar
    import pandas as pd
    data = {"a": ["5 4 3 2 1", "5 4 3 7 1", "6 7 7", "3 5 6 1", "1 2 3 4 5 6 7 8"], "b" :  [(3,2), (2,3), (1,2), (2,2), (2,3,2)]}
    df = pd.DataFrame(data)
    print df
    #df['z'] = df['a'].apply(getRadarLength)
    #print df[[1,0]].apply(separate_listInColumn, axis=1)
    #df['z'] = df[[1,0]].apply(separate_listInColumn, axis=1)
    #print df[['b','a']].apply(separate_listInColumn, axis=1)
    #df['r1'], df['r2'] = zip(*df[['b','a']].apply(separate_listInColumn, axis=1))
    #print '\n\n\n'
    #print df
    df['a1'] =  df[['b','a']].apply(getIthRadar, axis=1)
    print list(df['a1'])
    #print zip(*df['a'].apply(getListReductions))
    
if __name__ == "__main__":
    #print getRadarLength([5,4,3,2,1])
    #print getRadarLength([5,4,3,7,1])

    ##The following lines show how to create a new column with the length of each radar
    import pandas as pd
    data = {"a": ["5 4 3 2 1", "5 4 3 7 1", "6 7 7", "3 5 6 1", "1 2 3 4 5 6 7 8"], "b" :  [(3,2), (2,3), (1,2), (2,2), (2,3,2)]}
    df = pd.DataFrame(data)
    print df
    #df['z'] = df['a'].apply(getRadarLength)
    #print df[[1,0]].apply(separate_listInColumn, axis=1)
    #df['z'] = df[[1,0]].apply(separate_listInColumn, axis=1)
    #print df[['b','a']].apply(separate_listInColumn, axis=1)
    #df['r1'], df['r2'] = zip(*df[['b','a']].apply(separate_listInColumn, axis=1))
    #print '\n\n\n'
    #print df
    df['a1'] =  df[['b','a']].apply(getIthRadar, axis=1)
    print list(df['a1'])
    #print zip(*df['a'].apply(getListReductions))
