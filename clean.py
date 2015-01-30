def getRadarLength(TimeToEnd):
    ## Returns a n-tuble with (n1, n2...)
    ## where n? is the number of measurements for radar ?
    ## It is forced to be at least two since there should be max two radars...
    ## To add a dataframe column do:
    ## df['RadarLength'] = df['TimeToEnd'].apply(getRadarLength)
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
    if len(nlist)<2:
        nlist.append(0)
    return tuple(nlist)

def separate_listInColumn(x):
    ## First column must be the tuble of radar length
    ## Following columns should be columns to separate
    ## For now it gets only the first two radars
    # First translate the list string into a float string
    listrads = map(float,  x.iloc[1].split())
    # The list in then sliced by radar given in the first elements of x
    # x.iloc[0] is a tuple with the length of each radar measurement, i.e.
    # x.iloc[0][0] is the # of measurement with the 1st radar (x.iloc[0][1] for the 2nd radar)
    # The following line could be rewritten more clearly (by less efficient?) like this:
    # rad_measurements = x.iloc[0]
    # nrad1, nrad2 = x.iloc[0]
    # rad1, rad2 = listrads[:nrad1], listrads[nrad1:nrad1 + nrad2]
    return listrads[:x.iloc[0][0]], listrads[x.iloc[0][0]:x.iloc[0][0] + x.iloc[0][1]]

    
    
if __name__ == "__main__":
    #print getRadarLength([5,4,3,2,1])
    #print getRadarLength([5,4,3,7,1])

    ##The following lines show how to create a new column with the length of each radar
    import pandas as pd
    data = {"a": ["5 4 3 2 1", "5 4 3 7 1", "6 7 7", "3 5 6 1"], "b" :  [(3,2), (2,3), (1,2), (4,0)]}
    df = pd.DataFrame(data)
    print df
    #df['z'] = df['a'].apply(getRadarLength)
    #print df[[1,0]].apply(separate_listInColumn, axis=1)
    #df['z'] = df[[1,0]].apply(separate_listInColumn, axis=1)
    df['r1'], df['r2'] = zip(*df[['b','a']].apply(separate_listInColumn, axis=1))
    print '\n\n\n'
    print df
