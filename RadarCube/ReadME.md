## Information in case Range-Angle/Cross-Range plot will be implemted

in case the Cross-Range plot will be implemented here is some information and code you need to have.

Field of view in Azimuth is +-80°
For cross-range plot only the data from the azimuth antennas are taken into account. 
To realise this i provide the antenna array and a small script to only get the indices in radar cube 3D where azimuth data is located.

## Script
### specify filepath to AntennaArray
filepath_AntennaArray=''
### Antenna Array as 192x2 double array (represents the virtual array)
AntennaArray=np.load(filepath_AntennaArray)

### Get Antenna Array with only Azimuth Antenna receive combinations
ind=np.where(AntennaArray[:,1]== 0)
ind=np.asarray(ind)
ind=np.swapaxes(ind,0,1)
AntennaSelected=AntennaArray[ind,0]
val, indices= np.unique(AntennaSelected, return_index=True) 
AzimuthAntennaOnly= indices