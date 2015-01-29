def getRadarLength(TimeToEnd):
    ##Returns a n-tuble with (n1, n2...)
    ##where n? is the number of measurements for radar ?
    ##It is forced to be at least two since there should be max two radars...
    ##To add a dataframe column do:
    ##df['RadarLength'] = df['TimeToEnd'].apply(getRadarLength)
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

def test(x):
    print x

if __name__ == "__main__":
    #print getRadarLength([5,4,3,2,1])
    #print getRadarLength([5,4,3,7,1])

    ##The following lines show how to create a new column with the length of each radar
    import pandas as pd
    data = {"a": ["5 4 3 2 1", "5 4 3 7 1", "6 7 7"], "b" :  [1, 2, 6]}
    df = pd.DataFrame(data)
    print df
    df['z'] = df['a'].apply(getRadarLength)
    print '\n\n\n'
    print df
