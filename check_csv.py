import pandas as pd

print('csv check started')
def read_csv_withheaders(csv_name):
    with open(csv_name, "r") as f:
        time_series_list = pd.DataFrame(pd.read_csv(f, sep=";", header=0))
        return time_series_list


timeserieslist = read_csv_withheaders("csv_files/timeserieslist_config.csv")
rasterlist = read_csv_withheaders("csv_files/rasterlist_config.csv")
checklist = pd.concat([timeserieslist,rasterlist], axis=0)

if len(checklist.Type.unique()) != 6:
    raise ValueError("ERROR - Time series added with a wrong organization type") 

if len(checklist.naamlizard.unique()) != len(checklist):
    raise ValueError("ERROR - Time series added with an already existing name") 
    
# if len(timeserieslist.UUID.unique()) != len(timeserieslist):
#     raise ValueError("ERROR - Time series added with an already existing UUID") 

print('csv check completed')