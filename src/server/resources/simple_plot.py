import pandas as pd
import plotly.express as px
import math
import pandas as pd
import warnings
import plotly
warnings.filterwarnings("ignore")

#parks_df = pd.read_csv('../data/parks_data.csv')

def parks_plot(parks_df):

    parks_df = parks_df.dropna()

    def splitter(txt, delim):
        for i in txt:
            if i in delim:
                txt = txt.replace(i, ' ')
        return txt.split()

    parks_df_new = parks_df.copy()

    sep = ['(', ',', ')']

    parks_df_new['LATITUDE'] = 0
    parks_df_new['LONGITUDE'] = 0
    parks_df_new['NORM ACRES'] = 0

    i = 0
    parks_df_new = parks_df_new.drop([13, 32, 245, 371])
    parks_df_new = parks_df_new.reset_index(drop=True)

    while i < len(parks_df_new):
        parks_df_new['LATITUDE'][i] = float(splitter(parks_df_new['LOCATION'][i], sep)[-2])
        parks_df_new['LONGITUDE'][i] = float(splitter(parks_df_new['LOCATION'][i], sep)[-1])
        parks_df_new['NORM ACRES'][i] = math.sqrt(float(parks_df_new['ACRES'][i].replace(',', '')))
        parks_df_new['ACRES'][i] = float(parks_df_new['ACRES'][i].replace(',', ''))
        i += 1

    fig = px.scatter_mapbox(parks_df_new, lat=parks_df_new['LATITUDE'], lon=parks_df_new['LONGITUDE'],
                                        size=list(parks_df_new['NORM ACRES']),
                                        hover_name='LABEL', color="PARK CLASS", zoom=9.5, height=650)
    fig.update_layout(
                    mapbox_style='carto-positron')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()
    # print(plotly.io.to_json(fig))


parks_df = pd.read_csv('../data/parks_data.csv')
parks_plot(parks_df)
