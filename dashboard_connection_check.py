#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import requests
import json
import pathlib
import pathlib as pl
import numpy
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc

height = 750
lizard_color = "#3b3838"
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
APP_PATH = str(pl.Path(__file__).parent.resolve())
app = dash.Dash(__name__)
server = app.server

_app_route = "/"

app.config.suppress_callback_exceptions = True


username = "__key__"
password = "FS3cDFnh.QMGHUH8owuiSkezsV6SHfq6IkFe0YWeO"

headers = {"username": username, "password": password}


def create_lizard_api_timestamp(n, datem):  # n is the number of days to look back
    timenow = datem
    time_ago = datem - timedelta(days=n)
    td = timenow - time_ago
    td_minutes = ((td.days) * 1440 + (td.seconds) // 60) // n
    times_to_check = pd.date_range(
        start=time_ago.strftime("%Y-%m-%d"),
        end=timenow.strftime("%Y-%m-%d"),
        periods=(n + 1),
    ).strftime("%Y-%m-%dT%H:00:00Z")
    times_nan = pd.date_range(
        start=time_ago.strftime("%Y-%m-%d"),
        end=timenow.strftime("%Y-%m-%d"),
        periods=(n + 1),
    ).strftime("%Y-%m-%d")
    return td_minutes, times_to_check, times_nan


td, times_to_check, times_nan = create_lizard_api_timestamp(7, date.today())


def read_csv_withheaders(csv_name):
    with open(csv_name, "r") as f:
        time_series_list = pd.DataFrame(pd.read_csv(f, sep=";", header=0))
        return time_series_list


timeserieslist = read_csv_withheaders("csv_files/timeserieslist_config.csv")
rasterlist = read_csv_withheaders("csv_files/rasterlist_config.csv")
checklist = pd.concat([timeserieslist,rasterlist], axis=0)

def count_to_be(td_minutes, time_series_list):
    count_to_be = {}
    for index, row in time_series_list.iterrows():
        count_to_be[row["naamlizard"]] = td_minutes // row["interval"]
    #     count_to_be = pd.DataFrame(count_to_be.keys())
    return count_to_be


counttobe = {**count_to_be(td, timeserieslist), **count_to_be(td, rasterlist)}


def get_daily_counts(timeserieslist, times_to_check, times_nan):
    daily_counts = {}
    events_url = "https://nens.lizard.net/api/v4/timeseries/"
    days_to_check = [dt.split("T")[0] for dt in times_to_check ]
    for index, row in timeserieslist.iterrows():
        get_url = f"{events_url}{row['UUID']}/aggregates/"
        response = requests.get(
            url=get_url,
            headers=headers,
            params={
                "start": str(times_to_check[0]),
                "end": str(times_to_check[-1]),
                "fields": "count,first_timestamp",
                "window": "day",
            },
        )

        
        try:
            df = pd.DataFrame(response.json()["results"])
            df=df[df["first_timestamp"] == df["first_timestamp"]] #nanfilter
            df.set_index("first_timestamp", inplace=True)
            df.index = df.index.map(lambda s: s.split("T")[0])
            #Disregard last day as we notate the first timestamp of each period
            df = df[df.index.isin(days_to_check[0:-1])]
            
        except:
            df = pd.DataFrame(index=times_nan[0:-1], columns=["count"])
        daily_counts[row["naamlizard"]] = df
        
    df_counts = daily_counts[timeserieslist.naamlizard[0]].copy()
    df_counts.rename(columns={"count": timeserieslist.naamlizard[0]}, inplace=True)
    for i in range(1, len(timeserieslist)):
        df_counts[timeserieslist.naamlizard[i]] = daily_counts[
            timeserieslist.naamlizard[i]
        ]["count"]
    df_counts.replace(np.NaN, 0, inplace=True)
    return df_counts

def get_daily_counts_rast(rasterlist, times_to_check, times_nan):
    daily_counts = {}
    events_url = "https://nens.lizard.net/api/v4/rasters/"
    days_to_check = [dt.split("T")[0] for dt in times_to_check ]
    for index, row in rasterlist.iterrows():
        get_url = f"{events_url}{row['UUID']}/point"
        geom = "POINT({} {})".format(row["x"],row["y"])
        response = requests.get(
            url=get_url,
            headers=headers,
            params={
                "start": str(times_to_check[0]),
                "stop": str(times_to_check[-1]),
                "geom": geom,
                "frequency": "D",
                "statistic": "count",
                "label": "left"
            }
        )
        
        try:
            df = pd.DataFrame(response.json()["results"])
            df = df[df["time"] == df["time"]] #nanfilter
            df.set_index("time", inplace=True)
            df.index = df.index.map(lambda s: s.split("T")[0])
            #Disregard last timestep in times_to_check, as we use label left
            df = df[df.index.isin(days_to_check[0:-1])]
            
        except:
            df = pd.DataFrame(index=times_nan[0:-1], columns=["value"])
        daily_counts[row["naamlizard"]] = df
        
    df_counts = daily_counts[rasterlist.naamlizard[0]].copy()
    df_counts.rename(columns={"value": rasterlist.naamlizard[0]}, inplace=True)
    for i in range(1, len(rasterlist)):
        df_counts[rasterlist.naamlizard[i]] = daily_counts[
            rasterlist.naamlizard[i]
        ]["value"]
    df_counts.replace(np.NaN, 0, inplace=True)
    return df_counts

def data_availability(counts_timeseries, counts_rasters, count_to_be):
    data_availability = pd.concat([counts_timeseries, counts_rasters],axis=1)
    
    for col in data_availability:
        data_availability[col] = 100 * data_availability[col] / count_to_be[col]
    
    percentage_availability_average = pd.DataFrame(
        numpy.mean(data_availability)
    )
    return data_availability, percentage_availability_average

dict_waterboard = {}
dict_province = {}
dict_drinking = {}
dict_country = {}
dict_municipality = {}
dict_g4aw = {}
waterboards = checklist.loc[checklist.Type == "Waterboard"]
waterboards_unique = waterboards.Organization.unique()
provinces = checklist.loc[checklist.Type == "Province"]
provinces_unique = provinces.Organization.unique()
countrywides = checklist.loc[checklist.Type == "Country"]
countrywides_unique = countrywides.Organization.unique()
drinkings = checklist.loc[checklist.Type == "Drinking"]
drinkings_unique = drinkings.Organization.unique()
municipalitys = checklist.loc[checklist.Type == "Municipality"]
municipalitys_unique = municipalitys.Organization.unique()
g4aws = checklist.loc[checklist.Type == "G4AW"]
g4aws_unique = g4aws.Organization.unique()

for waterboard in waterboards_unique:
    dict_waterboard[waterboard] = waterboards.loc[
        waterboards.Organization == (waterboard)
    ]
for province in provinces_unique:
    dict_province[province] = provinces.loc[provinces.Organization == (province)]
for countrywide in countrywides_unique:
    dict_country[countrywide] = countrywides.loc[
        countrywides.Organization == (countrywide)
    ]
for drinking in drinkings_unique:
    dict_drinking[drinking] = drinkings.loc[drinkings.Organization == (drinking)]
for municipality in municipalitys_unique:
    dict_municipality[municipality] = municipalitys.loc[
        municipalitys.Organization == (municipality)
    ]
for municipality in municipalitys_unique:
    dict_municipality[municipality] = municipalitys.loc[
        municipalitys.Organization == (municipality)
    ]
for g4aw in g4aws_unique:
    dict_g4aw[g4aw] = g4aws.loc[g4aws.Organization == (g4aw)]



def connection_status_overal(percentage_availability_daily, checklist):
    con_stat = []
    con_stat = pd.DataFrame(columns=["Type", "Available"])
    i = 0
    type_unique = checklist.Type.unique()
    for typ in type_unique:
        con_stat.loc[i] = [
            typ,
            percentage_availability_daily[
                checklist.loc[checklist.Type == typ].naamlizard
            ].values.mean(),
        ]
        i = i + 1
    con_stat["Missing"] = 100 - con_stat["Available"]
    return con_stat

#%%

def figure_overall_connection(con_stat):
    labels = ["Available", "Missing"]
    colors = ["green", "red"]
    fig = make_subplots(
        rows=1,
        cols=len(con_stat),
        specs=[
            [
                {"type": "domain"},
                {"type": "domain"},
                {"type": "domain"},
                {"type": "domain"},
                {"type": "domain"},
                {"type": "domain"},
            ]
        ],
        subplot_titles=con_stat["Type"].values,
    )

    for i in range(len(con_stat)):
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=[con_stat.Available[i], con_stat.Missing[i]],
                name=con_stat["Type"][i],
            ),
            1,
            i + 1,
        )
    fig.update_traces(
        marker=dict(colors=colors, line=dict(color="#000000", width=2)),
        hole=0.4,
        hoverinfo="label+percent+name",
    )
    #     fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
    #                   marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig.update_layout(height=350)
    fig.update_layout(width=1000)
    return fig


def select_type_data(percentage_availability_daily, timeserieslist, type_selection):
    df_selected_type = percentage_availability_daily[
        timeserieslist.loc[timeserieslist.Type == type_selection].naamlizard
    ]
    df_selected_type_organization = []
    if type_selection == "Waterboard":
        dict_selected = dict_waterboard
        type_selected = waterboards
    elif type_selection == "Province":
        dict_selected = dict_province
        type_selected = provinces
    elif type_selection == "Drinking":
        dict_selected = dict_drinking
        type_selected = drinkings
    elif type_selection == "Country":
        dict_selected = dict_country
        type_selected = countrywides
    elif type_selection == "Municipality":
        dict_selected = dict_municipality
        type_selected = municipalitys
    elif type_selection == "G4AW":
        dict_selected = dict_g4aw
        type_selected = g4aws       
    for i in range(len((type_selected.Organization.unique()))):
        df_selected_type[type_selected.Organization.unique()[i]] = (
            df_selected_type[
                dict_selected[type_selected.Organization.unique()[i]].naamlizard
            ]
        ).mean(axis=1)
    df_selected_type_organization = df_selected_type[
        (type_selected.Organization.unique())
    ]
    df = df_selected_type_organization
    return df


def figure_type(df_selected_type_organization):
    fig = make_subplots(
        rows=len(df_selected_type_organization.columns), cols=1, shared_xaxes=True
    )
    for i in range(len(df_selected_type_organization.columns)):
        fig.add_trace(
            go.Bar(
                x=df_selected_type_organization.index,
                y=round(
                    df_selected_type_organization[
                        df_selected_type_organization.columns[i]
                    ],
                    0,
                ),
                marker={
                    "color": df_selected_type_organization[
                        df_selected_type_organization.columns[i]
                    ],
                    "colorscale": "RdYlGn",
                    "cmin": 0,
                    "cmax": 100,
                },
                text=round(
                    df_selected_type_organization[
                        df_selected_type_organization.columns[i]
                    ],
                    0,
                ),
                textposition="auto",
                name=df_selected_type_organization.columns[i],
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(showlegend=False, height=1000, width=1000)
    for i in range(len(df_selected_type_organization.columns)):
        fig.update_yaxes(
            title_text=df_selected_type_organization.columns[i], row=i + 1, col=1
        )
    return fig


def select_organization_source(
    percentage_availability_daily, timeserieslist, type_selection, selectionfromfigure
):
    if type_selection == "Waterboard":
        dict_selected = dict_waterboard
        uniq_selected = waterboards_unique
    elif type_selection == "Province":
        dict_selected = dict_province
        uniq_selected = provinces_unique
    elif type_selection == "Drinking":
        dict_selected = dict_drinking
        uniq_selected = drinkings_unique
    elif type_selection == "Country":
        dict_selected = dict_country
        uniq_selected = countrywides_unique
    elif type_selection == "Municipality":
        dict_selected = dict_municipality
        uniq_selected = municipalitys_unique
    elif type_selection == "G4AW":
         dict_selected = dict_g4aw
         uniq_selected = g4aws_unique       
    selected_organization = uniq_selected[selectionfromfigure]
    df_selected_type = percentage_availability_daily[
        dict_selected[(selected_organization)].naamlizard
    ]
    df_selected_type_organization = []
    for i in range(len((dict_selected[(selected_organization)].Source.unique()))):
        df_selected_type[
            (dict_selected[(selected_organization)].Source.unique())[i]
        ] = (
            percentage_availability_daily[
                dict_selected[(selected_organization)][
                    dict_selected[(selected_organization)]["Source"]
                    == (dict_selected[str(selected_organization)].Source.unique())[i]
                ].naamlizard
            ]
        ).mean(
            axis=1
        )
    df_selected_type_organization = df_selected_type[
        (dict_selected[(selected_organization)].Source.unique())
    ]
    df = df_selected_type_organization
    #     df.reset_index(inplace = True)
    #     df.rename(columns={"index": "Source"}, inplace = True)
    return df


# --------------------------------------------------------------------------------------------------------------------------
# define dashboard layout
# Banner
banner = html.Div(
    className="pkcalc-banner",
    children=[
        html.Div(
            [
                html.Img(
                    src=app.get_asset_url("fewslogo.png"),
                    id="utrecht-wapen",
                    style={
                        "height": "80px",
                        "width": "auto",
                        "margin-bottom": "0px",
                        "vertical-align": "top",
                    },
                )
            ],
            style={"text-align": "center", "display": "inline-block"},
            className="one column",
            id="Utrechtlogo",
        ),
        html.Div(
            [
                html.Img(
                    src=app.get_asset_url("NenS_beeldmerk_dia_RGB.png"),
                    id="nenslogo",
                    style={
                        "height": "50px",
                        "width": "auto",
                        "margin-top": "0px",
                        "vertical-align": "center",
                    },
                )
            ],
            style={
                "text-align": "left",
                "display": "inline-block",
                "width": "60px",
            },
            className="one column",
            id="Nenslogo",
        ),
        html.Div(
            [
                html.Img(
                    src=app.get_asset_url("lizard-logo-white.png"),
                    id="plotly-image",
                    style={
                        "height": "34px",
                        "width": "auto",
                        "margin-top": "10px",
                        "vertical-align": "left",
                    },
                )
            ],
            style={"textAlign": "center", "vertical-align": "left"},
            className="one column",
        ),
    ],
    style={"height": "80px", "background-color": lizard_color},
)
href = "https://nens.lizard.net/nl/map/topography/point/@52.1858,5.2677,8/-2Days0Hours+0Days3Hours"

content = html.Div(
    [
        html.Div(
            [
                html.H5("Select Time", style={"font-weight": "bold"}),
                html.Br([]),
                html.P(
                    "\
                                                    A 7 day connection status report will be created until\
                                                    the selected date.\
                                                    ",
                    style={"color": "black"},
                    className="row",
                ),
                html.Br([]),
                dcc.DatePickerSingle(
                    id="my-date-picker-single",
                    min_date_allowed=date(2000, 1, 1),
                    max_date_allowed=None,
                    initial_visible_month=None,
                    date=None,
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=12 * 60 * 60 * 1000,  # in milliseconds
                    n_intervals=0,
                ),
                html.Br([]),
                html.Br([]),
                html.Div(
                    [],
                    style={"borderBottom": "1px solid #d6d6d6"},
                    className="divBorder",
                ),
                html.Br([]),
                html.Br([]),
                html.H5("Select Organization Type", style={"font-weight": "bold"}),
                html.Br([]),
                html.P(
                    "\
                                                    A summary of all organizations in the selected type\
                                                    will be shown.\
                                                    ",
                    style={"color": "black"},
                    className="row",
                ),
                html.Br([]),
                dcc.RadioItems(
                    id="organization_type",
                    options=[
                        {"label": "Waterboards", "value": "Waterboard"},
                        {"label": "Provinces", "value": "Province"},
                        {"label": "Municipalities", "value": "Municipality"},
                        {"label": "Countrywide", "value": "Country"},
                        {"label": "Drinking Water Companies", "value": "Drinking"},
                        {"label": "G4AW Projects", "value": "G4AW"},
                    ],
                    value="Waterboard",
                ),
                html.Br([]),
                html.Div(
                    [],
                    style={"borderBottom": "1px solid #d6d6d6"},
                    className="divBorder",
                ),
                html.Br([]),
                html.Br([]),
                html.H5("Info About the System", style={"font-weight": "bold"}),
                html.Br([]),
            ],
            # ## # ##
            className="pretty_container two columns",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5(
                                    "Overall connection status of all systems (per system type)",
                                    style={
                                        "text-align": "center",
                                        "font-size": "16px",
                                        "font-weight": "bold",
                                    },
                                ),
                                dcc.Graph(
                                    id="mapbox_figure3",
                                    figure={},
                                    style={
                                        "padding": "10px",
                                        "display": "inline-block",
                                    },
                                ),
                            ],
                            className="pretty_container",
                        ),
                    ],
                    className="row",
                ),
                dcc.Loading(
                    id="loading-1",
                    children=html.Div(
                        id="intermediate-value", style={"display": "none"}
                    ),
                    type="default",
                    fullscreen=True,
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                #                                             dcc.Graph(id='mapbox_figure3', figure={},
                                #                                                       style={"padding": "10px", 'display': 'inline-block'}),
                                html.H5(
                                    "Connection status of the organizations in the selected system type",
                                    style={
                                        "text-align": "center",
                                        "font-size": "16px",
                                        "font-weight": "bold",
                                    },
                                ),
                                html.H5(
                                    "Clicking on this figure will show more info about all data sources in that organization",
                                    style={"text-align": "center", "font-size": "10px"},
                                ),
                                dcc.Graph(
                                    id="sum_figure_1",
                                    figure={},
                                    hoverData={"points": [{"customdata": "1"}]},
                                    style={
                                        "padding": "10px",
                                        "display": "inline-block",
                                    },
                                ),
                            ],
                            className="pretty_container",
                        ),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5(
                                    "Connection status of the data sources in the selected organization",
                                    style={
                                        "text-align": "center",
                                        "font-size": "16px",
                                        "font-weight": "bold",
                                    },
                                ),
                                dcc.Graph(
                                    id="sum_figure_2",
                                    figure={},
                                    style={
                                        "padding": "10px",
                                        "display": "inline-block",
                                    },
                                ),
                            ],
                            className="pretty_container",
                        ),
                    ],
                    className="row",
                ),
            ],
            className="eight columns",
        ),
    ],
    className="row flex-display",
    style={"padding": 0},
)

app.layout = html.Div(
    [
        banner,
        html.Div(
            [],
            style={"borderBottom": "1px solid #d6d6d6"},
            className="divBorder",
        ),
        html.Br([]),
        html.Br([]),
        html.Div(
            [
                html.P(
                    "    ",
                    className="eleven columns",
                    style={"text-align": "center", "heigt": "100px"},
                )
            ],
            className="row flex-display",
        ),
        html.H5(
            "FEWS - Lizard Connection Status Dashboard",
            style={"text-align": "center", "font-size": "32px", "font-weight": "bold"},
        ),
        html.Div(
            [
                html.P(
                    "    ",
                    className="eleven columns",
                    style={
                        "text-align": "center",
                        "font-size": "32px",
                        "font-weight": "bold",
                    },
                ),
                html.Br([]),
            ],
            className="row flex-display",
        ),
        html.Br([]),
        html.Div(
            [],
            style={"borderBottom": "1px solid #d6d6d6"},
            className="divBorder",
        ),
        html.Br([]),
        #                        sidebar,
        content,
    ]
)


@app.callback(
    dash.dependencies.Output("my-date-picker-single", "max_date_allowed"),
    dash.dependencies.Output("my-date-picker-single", "initial_visible_month"),
    dash.dependencies.Output("my-date-picker-single", "date"),
    [dash.dependencies.Input("interval-component", "n_intervals")],
)
def updatedate(input):
    return date.today(), date.today(), date.today()


@app.callback(
    dash.dependencies.Output(
        component_id="intermediate-value", component_property="children"
    ),
    [
        dash.dependencies.Input(
            component_id="my-date-picker-single", component_property="date"
        )
    ],
)
def get_data_from_lizard(date_value):
    start = datetime.strptime(date_value, "%Y-%m-%d")
    td, times_to_check, times_nan = create_lizard_api_timestamp(7, start)
    counts_ts = get_daily_counts(timeserieslist, times_to_check, times_nan)
    counts_rast = get_daily_counts_rast(rasterlist, times_to_check, times_nan)
    percentage_availability_daily, percentage_availability_average = data_availability(
        counts_ts, counts_rast, counttobe
    )
    datasets_rf = {
        "percentage_availability_daily": percentage_availability_daily.to_json(
            orient="split", date_format="iso"
        ),
    }

    return json.dumps(datasets_rf)


@app.callback(
    dash.dependencies.Output(
        component_id="mapbox_figure3", component_property="figure"
    ),
    [
        dash.dependencies.Input(
            component_id="intermediate-value", component_property="children"
        )
    ],
)
def update_overal_connection(jsonified_cleaned_data):
    datasets_rf = json.loads(jsonified_cleaned_data)
    percentage_availability_daily = pd.read_json(
        datasets_rf["percentage_availability_daily"], orient="split"
    )
    con_stat = connection_status_overal(percentage_availability_daily, checklist)
    figure = figure_overall_connection(con_stat)
    return figure


@app.callback(
    dash.dependencies.Output(component_id="sum_figure_1", component_property="figure"),
    [
        dash.dependencies.Input(
            component_id="organization_type", component_property="value"
        ),
        dash.dependencies.Input(
            component_id="intermediate-value", component_property="children"
        ),
    ],
)
def update_per_organization(organization_type, jsonified_cleaned_data):
    datasets_rf = json.loads(jsonified_cleaned_data)
    percentage_availability_daily = pd.read_json(
        datasets_rf["percentage_availability_daily"], orient="split"
    )
    df_selected_type_organization = select_type_data(
        percentage_availability_daily, checklist, organization_type
    )
    figure = figure_type(df_selected_type_organization)
    return figure


@app.callback(
    dash.dependencies.Output(component_id="sum_figure_2", component_property="figure"),
    [
        dash.dependencies.Input(
            component_id="organization_type", component_property="value"
        ),
        dash.dependencies.Input(
            component_id="intermediate-value", component_property="children"
        ),
        dash.dependencies.Input("sum_figure_1", "clickData"),
    ],
)
def display_click_data(organization_type, jsonified_cleaned_data, clickData):
    if clickData != None:
        selectionfromfigure = clickData["points"][0]["curveNumber"]
        datasets_rf = json.loads(jsonified_cleaned_data)
        percentage_availability_daily = pd.read_json(
            datasets_rf["percentage_availability_daily"], orient="split"
        )
        df_selected_organization = select_organization_source(
            percentage_availability_daily,
            timeserieslist,
            organization_type,
            int(selectionfromfigure),
        )
        figure = figure_type(df_selected_organization)
    else:
        figure = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"secondary_y": True}]],
            column_widths=[1],
            row_heights=[0.4],
        )
    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
