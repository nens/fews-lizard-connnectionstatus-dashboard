# Dashboard Fews-Lizard Connection Status

This is a dashboard for checking the data in lizard for different organizations by default last 7 days. 

Project code : W0001.6 - Delft Fews

Dashboard is configured from the timeserieslist_config.csv file in csv_files folder. Available organizations types are 

- Waterboard

- Province

- Country

- Drinking

- Municipality

You can add new time series for different organizations/ different sources to the csv file. 
**Check for spelling if an organization, source already exists in the csv file.**

We added a csv_check.py that should rise an error if there is a problem with the csv file. 

A new timeseries entry should have :

Organization || Type || Source || UUID || interval || naamlizard

Main Script uses an APIkey of the account: **sa_monitoring**

If a new organization is added to the .csv, read rights of that organization has to be asked to the account.

