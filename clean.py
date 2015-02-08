import numpy as N

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
    '''
    Returns a list of measurements for the ith radar
    Returns None if there are no ith radar
    Input:
        x : should be a panda Dataframe
                - First column must be the tuble of radar length
                (as produced by getRadarLength)
                - Second columns must contains the values to separate
        iradar : the ith radar to return the data from (default=1st)
    '''
    if len(x.iloc[0])<iradar:
        return None## Or should it be NA?
    ## The longer but clearer way
    listrads = map(float,  x.iloc[1].split())
    rad_start_index = sum(x.iloc[0][:iradar-1])
    rad_stop_index = rad_start_index + x.iloc[0][iradar-1]
    return tuple(listrads[rad_start_index:rad_stop_index])

def getListReductions(x):
    '''
    Returns mean, range (max - min) and # of values of the given list
    Input:
        x : should be a list or tuple
        (or something that can be turned into a numpy array)
    '''
    xarray = N.array(x)
    return xarray.mean(), xarray.ptp(axis=0), len(xarray)

def getStringReductions(x):
    '''
    Returns mean, range (max - min) and # of values of the given string
    Input:
        x : should be a space-separated value string
    '''
    xarray = N.array(map(float, x.split()))
    return xarray.mean(), xarray.ptp(axis=0), len(xarray)

    
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
