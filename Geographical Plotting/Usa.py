import chart_studio.plotly as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
data = dict(type='choropleth',
            location=['AZ','CA','NY'], 
            locationmode='USA-states',
            colorscale='Portland', 
            text = ['text 1', 'text 2', 'text 3'],
            z = [1.0,2.0,3.0], 
            colorbar = {'title':'Colorbar Title Goes here'})
layout = dict(geo = {'scope':'usa'})
choromap = go.Figure(data = [data], layout=layout)
iplot(choromap)