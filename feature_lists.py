"""Lists of features to fit."""

hm_types = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]

list1 = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
            'Avg_DistanceToRadar', 'Avg_RadarQualityIndex',
            'Range_RadarQualityIndex',
            'Avg_RR1', 'Range_RR1', 'Avg_RR2', 'Range_RR2',
            'Avg_RR3', 'Range_RR3', 'Avg_Zdr', 'Range_Zdr',
            'Avg_Composite', 'Range_Composite', 'Avg_HybridScan',
            'Range_HybridScan', 'Avg_Velocity', 'Range_Velocity',
            'Avg_LogWaterVolume', 'Range_LogWaterVolume',
            'Avg_MassWeightedMean', 'Range_MassWeightedMean',
            'Avg_MassWeightedSD', 'Range_MassWeightedSD',
            'Avg_RhoHV', 'Range_RhoHV', 'Avg_Kdp', 'Range_Kdp',
            ]
list1.extend(["hm_{}".format(i) for i in hm_types])


def get_list1():
    """Return a copy a the list so original is not changed."""
    return list(list1)
