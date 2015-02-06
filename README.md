# krad
understanding polarized radar data

## Data
The raw data comes from a csv file which comes from two radars but often only one made measurements

## Variables

* TimeToEnd:  How many minutes before the end of the hour was this radar observation?

* DistanceToRadar:  Distance between radar and gauge.  This value is scaled and rounded to prevent reverse engineering gauge location

* Composite:  Maximum reflectivity in vertical volume above gauge

* HybridScan: Reflectivity in elevation scan closest to ground

* HydrometeorType:  One of nine categories in NSSL HCA. See presentation for details.

* Kdp:  Differential phase

* RR1:  Rain rate from HCA-based algorithm

* RR2:  Rain rate from Zdr-based algorithm

* RR3:  Rain rate from Kdp-based algorithm

* RadarQualityIndex:  A value from 0 (bad data) to 1 (good data)

* Reflectivity:  In dBZ see notebook reflectivityAvgNrange.ipynb

* ReflectivityQC:  Quality-controlled reflectivity

* RhoHV:  Correlation coefficient

* Velocity:  (aliased) Doppler velocity

* Zdr:  Differential reflectivity in dB

* LogWaterVolume:  How much of radar pixel is filled with water droplets?

* MassWeightedMean:  Mean drop size in mm

* MassWeightedSD:  Standard deviation of drop size

* Expected: the actual amount of rain reported by the rain gauge for that hour.

