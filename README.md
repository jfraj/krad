# krad
understanding polarized radar data

## Data
The raw data comes from a csv file which comes from a variable number of radar

## Variables
Info comes from from the competition's [data page](https://www.kaggle.com/c/how-much-did-it-rain/data) and from this [training](http://www.wdtb.noaa.gov/courses/dualpol/).

### Unpolarized
* **TimeToEnd**:  How many minutes before the end of the hour was this radar observation?
* **DistanceToRadar**:  Distance between radar and gauge.  This value is scaled and rounded to prevent reverse engineering gauge location (see notebook distancetoradar.ipynb for more information)
* **Composite**:  Maximum reflectivity in vertical volume above gauge
* **HybridScan**: Reflectivity in elevation scan closest to ground
* **Reflectivity**:  In dBZ see notebook reflectivityAvgNrange.ipynb
* **ReflectivityQC**:  Quality-controlled reflectivity
* **Velocity**:  (aliased) Doppler velocity
* **LogWaterVolume**:  How much of radar pixel is filled with water droplets?
* **MassWeightedMean**:  Mean drop size in mm
* **MassWeightedSD**:  Standard deviation of drop size
* **RR1**:  Rain rate from HCA-based algorithm (HCA: Hydrometeor Classification Algorithm)
* **Expected**: the actual amount of rain reported by the rain gauge for that hour.

### Polarized
* **Zdr**:  Differential reflectivity in dB.  Horizontal - Vertical reflectivity.
* **Kdp**:  Specific differential phase.  It's a _local_ variable, defined as the correlation coefficient slope (as calculated at two distance ranges).  Note: KDP is "displayed" (calculated?) only when RhoHV>0.9.
* **RR2**:  Rain rate from Zdr-based algorithm
* **RR3**:  Rain rate from Kdp-based algorithm
* **RhoHV**:  Correlation coefficient.  Normalized sum of all v(hpol) - v(vpol) in the over all the radar pulses.(hpol/vpol are horizontal/vertical polarization).  Should be between 0,1 but above 1 is related to noise it should _not_ be trunckated to 1.  Pure rain should be very close to 1, while many different scatteres ~0 (i.e. no correlations).  Effectively, RhoHV<0.8 is so uncorrlated, it usually not meteorological e.g. birds.  RhoHV>0.97 is usually _pure_ rain or snow.

### ?polarized (TBD)
* **RadarQualityIndex**:  A value from 0 (bad data) to 1 (good data)
