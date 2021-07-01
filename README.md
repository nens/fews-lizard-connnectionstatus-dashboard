# Dashboard Fews-Lizard Connection Status

This is a dashboard for checking the data in lizard for different organizations by default last 7 days. 

Project code : W0001.6 - Delft Fews

Dashboard is configured from the timeserieslist_config.csv and rasterlist_config.csv files in csv_files folder. 

Available organizations types are 

- Waterboard

- Province

- Country

- Drinking

- Municipality

- G4AW

You can add new time series for different organizations/ different sources to the csv files. 
**Check for spelling if an organization, source already exists in a csv file.**

We added a csv_check.py that should rise an error if there is a problem with the csv files. 

A new timeseries entry should have :

Organization || Type || Source || UUID || interval || naamlizard

A new raster entry should have:

Organization || Type || Source || UUID || interval || naamlizard  || x-coordinate || y-coordinate

The coordinates in raster entries are used for the API request, so they should describe a point where the raster is expected to have data values. 

Main Script uses an APIkey of the account: **sa_monitoring**

If a new organization is added to the .csv, read rights of that organization has to be asked to the account.

