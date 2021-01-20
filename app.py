# Import Modules
from base64 import b64encode
from flask import Flask, render_template, request, redirect, url_for, session, make_response
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from io import StringIO
import io
import csv
import os
import numpy as np
import pandas as pd
from helper_v4 import determine_timeframe,get_summary_stats,preprocessing, multiForecast
import logging
from sqlalchemy import create_engine
import psycopg2
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool

LOCAL = False

if(LOCAL):
    db = 'postgresql://postgres:Michigan123@localhost:5433/forecast_ai'
else:
    db = 'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai'

# Socket IO Flask App Setup

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['MAX_CONTENT_LENGTH'] = 1000* 1024 * 1024
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
if(LOCAL):
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Michigan123@localhost:5433/forecast_ai'
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai'
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

socketio = SocketIO(app, logger=False, engineio_logger=False, ping_timeout=3600)
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), unique=True, nullable=False)


    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __repr__(self):
        return '<User {}>'.format(self.username)

class Data(db.Model):
    id = db.Column(db.String(120), primary_key=True)
    username = db.Column(db.String(80))
    series_id = db.Column(db.String(100))
    date = db.Column(db.Date())
    value = db.Column(db.Float())


    def __init__(self, username, series_id, date, value):
        self.username = username
        self.series_id = series_id
        self.date = date
        self.value = value

    def __repr__(self):
        return '<User {}>'.format(self.username)

class Metrics(db.Model):
    id = db.Column(db.String(120), primary_key=True)
    username = db.Column(db.String(80))
    series_id = db.Column(db.String(100))
    date = db.Column(db.Date())
    residual = db.Column(db.Float())
    absolute_error = db.Column(db.Float())
    squared_error = db.Column(db.Float())
    benchmark_residual = db.Column(db.Float())


    def __init__(self, username, series_id, date, residual, absolute_error, squared_error, benchmark_residual):
        self.username = username
        self.series_id = series_id
        self.date = date
        self.residual = residual
        self.absolute_error = absolute_error
        self.squared_error = squared_error
        self.benchmark_residual = benchmark_residual

    def __repr__(self):
        return '<User {}>'.format(self.username)

class Stats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    series_id = db.Column(db.String(100))
    min = db.Column(db.Float())
    max = db.Column(db.Float())
    mean = db.Column(db.Float())
    sum = db.Column(db.Float())


    def __init__(self, username, series_id, min, max, mean, sum):
        self.username = username
        self.series_id = series_id
        self.min = min
        self.max = max
        self.mean = mean
        self.sum = sum

    def __repr__(self):
        return '<User {}>'.format(self.username)

class Output(db.Model):
    id = db.Column(db.String(120), primary_key=True)
    username = db.Column(db.String(80))
    series_id = db.Column(db.String(100))
    date = db.Column(db.Date())
    value = db.Column(db.Float())
    forecast = db.Column(db.Float())


    def __init__(self, username, series_id, date, value, forecast):
        self.username = username
        self.series_id = series_id
        self.date = date
        self.value = value
        self.forecast = forecast

    def __repr__(self):
        return '<User {}>'.format(self.username)

class StandardOutput(db.Model):
    id = db.Column(db.String(120), primary_key=True)
    username = db.Column(db.String(80))
    series_id = db.Column(db.String(100))
    last_total = db.Column(db.Float())
    next_total = db.Column(db.Float())
    change = db.Column(db.String(30))


    def __init__(self, username, series_id, last_total, next_total, change):
        self.username = username
        self.series_id = series_id
        self.last_total = last_total
        self.next_total = next_total
        self.change = change


    def __repr__(self):
        return '<User {}>'.format(self.username)


# Suppress logs except for error: https://stackoverflow.com/questions/43487264/disabling-logger-in-flask-socket-io
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('geventwebsocket.handler').setLevel(logging.ERROR)



@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# Flask App
@app.route('/post_user', methods=['POST','GET'])
def post_user():
    user = User(request.form['username'], request.form['password'])
    login = 0
    for us in User.query.all():
        if(us.username == user.username):
            if(us.password == user.password):
                login=1
    if(login==1):
        print("User Logged In")
        session["username"] = request.form['username']

        return redirect(url_for('data'))
    else:
        print("Creating New User")
        db.session.add(user)
        db.session.commit()
        session["username"] = request.form['username']
        return redirect(url_for('data'))

@app.route("/export-detailed")
def csv_export_detailed():
    print('Downloading Details')
    if(LOCAL):
        engine = create_engine("postgresql://postgres:Michigan123@localhost:5433/forecast_ai")
    else:
        engine = create_engine('postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai')
    query = "select * from output where username = '{}'".format(session['username'])
    data = pd.read_sql(query, engine)
    data.drop(['id','username'], axis=1, inplace=True)

    output = [data.columns.tolist()] + data.values.tolist()
    data = output


    si = StringIO()
    cw = csv.writer(si)
    cw.writerows(data)
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=detailed_ouput.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route("/export-standard")
def csv_export_standard():
    print('Downloading Standard')
    if (LOCAL):
        engine = create_engine("postgresql://postgres:Michigan123@localhost:5433/forecast_ai")
    else:
        engine = create_engine(
            'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai')
    query = "select * from standard_output where username = '{}'".format(session['username'])
    data = pd.read_sql(query, engine)
    data.drop(['id','username'], axis=1, inplace=True)

    output = [data.columns.tolist()] + data.values.tolist()
    data = output


    si = StringIO()
    cw = csv.writer(si)
    cw.writerows(data)
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=standard_output.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@app.route('/workflow/', methods=['GET', 'POST'])
def data():
    if (LOCAL):
        engine = create_engine("postgresql://postgres:Michigan123@localhost:5433/forecast_ai")
    else:
        engine = create_engine(
            'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai')
    query = "select series_id, date, value from data where username = '{}'".format(session['username'])
    df = pd.read_sql(query, engine)
    df = df.groupby(['date'], as_index=False)['value'].sum()

    TOOLS="pan,wheel_zoom,box_zoom,reset,save"
    source = ColumnDataSource(data={
        'date': df['date'],
        'value': df['value'],
    })

    p = figure(plot_width=800, plot_height=500, x_axis_type="datetime", tools=TOOLS)
    p.background_fill_color = "#f5f5f5"
    p.grid.grid_line_color = "white"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Value'
    p.axis.axis_line_color = None

    p.line(x='date', y='value', line_width=2, color='#1a8ae5', source=source)
    #p.show(date_range_slider)

    p.add_tools(HoverTool(
        tooltips=[
            ('date', '@date{%F}'),
            ('value', '@{value}{%0.2f}'),  # use @{ } for field names with spaces
        ],

        formatters={
            'date': 'datetime',  # use 'datetime' formatter for 'date' field
            'value': 'printf',  # use 'printf' formatter for 'adj close' field
            # use default 'numeral' formatter for other fields
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline'
    ))

    script, div = components(p)

    ################################
    ## Setting up data statistics ##
    ################################
    if (LOCAL):
        engine = create_engine("postgresql://postgres:Michigan123@localhost:5433/forecast_ai")
    else:
        engine = create_engine(
            'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai')
    query = "select series_id, min, max, mean, sum from stats where username = '{}'".format(session['username'])
    df = pd.read_sql(query, engine)
    count = int(len(df))
    minval = np.round(np.min(df['min']),2)
    maxval = np.round(np.max(df['max']),2)
    meanval = np.round(np.mean(df['mean']),2)

    results = [['ID','Minimum','Maximum','Average','Sum']]
    for i in range(len(df)):
        results.append(df.values[i])

    if (LOCAL):
        engine = create_engine("postgresql://postgres:Michigan123@localhost:5433/forecast_ai")
    else:
        engine = create_engine(
            'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai')
    query = "select * from standard_output where username = '{}'".format(session['username'])
    df = pd.read_sql(query, engine)
    lasttotal = np.round(np.sum(df['last_total']),0)
    nexttotal = np.round(np.sum(df['next_total']),0)
    change = str(np.round(100*((nexttotal - lasttotal) / lasttotal),2)) + '%'
    lastmean = np.round(np.mean(df['last_total']), 2)
    nextmean = np.round(np.mean(df['next_total']), 2)

    query = "select * from metrics where username = '{}'".format(session['username'])
    df = pd.read_sql(query, engine)
    mae = np.round(np.mean(df['absolute_error']),2)
    rmse = np.round(np.sqrt(np.mean(df['squared_error'])),2)

    ################################
    ## Generating Forecast Graphs ##
    ################################
    if (LOCAL):
        engine = create_engine("postgresql://postgres:Michigan123@localhost:5433/forecast_ai")
    else:
        engine = create_engine(
            'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai')

    query = "select series_id, date, value, forecast from output where username = '{}'".format(session['username'])
    df = pd.read_sql(query, engine)
    fcdate = df[df.forecast.notnull()]['date'].min()
    maxdate = df[df.value.notnull()]['date'].max()
    df = df.groupby(['date'], as_index=False).agg({'value':'sum', 'forecast':'sum'})
    #df = df.replace(0, np.nan)
    df['forecast'][df.date < fcdate] = np.nan
    df['value'][df.date > maxdate] = np.nan
    source = ColumnDataSource(data={
        'date': df['date'],
        'value': df['value'],
        'forecast': df['forecast']
    })

    p = figure(plot_height=250, x_axis_type="datetime", tools=TOOLS, sizing_mode="scale_width")
    p.background_fill_color = "#f5f5f5"
    p.grid.grid_line_color = "white"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Value'
    p.axis.axis_line_color = None

    p.line(x='date', y='value', line_width=2, color='#1a8ae5', source=source)
    p.line(x='date', y='forecast', line_width=2, color='#eea411', source=source)

    p.add_tools(HoverTool(
        tooltips=[
            ('date', '@date{%F}'),
            ('value', '@{value}{%0.2f}'),  # use @{ } for field names with spaces
            ('forecast', '@{forecast}{%0.2f}'),  # use @{ } for field names with spaces
        ],

        formatters={
            'date': 'datetime',  # use 'datetime' formatter for 'date' field
            'value': 'printf',  # use 'printf' formatter for 'adj close' field
            'forecast': 'printf',  # use 'printf' formatter for 'adj close' field
            # use default 'numeral' formatter for other fields
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline'
    ))

    script_output, div_output = components(p)

    return render_template('workflow.html', username=session['username'], table=results,
                           count=count, minval=minval, maxval=maxval, meanval=meanval,
                           lasttotal=lasttotal, nexttotal=nexttotal, change=change,
                           lastmean=lastmean, nextmean=nextmean,
                           mae=mae, rmse=rmse, div=div, script=script,
                           script_output=script_output, div_output=div_output) # Application


@app.route('/app/')
def index():
    return render_template('build-forecast-v3.html') # Application

@app.route('/')
def about():
    return render_template('login.html')

@socketio.on('refresh')
def refresh_data():
    #messages = request.args['messages']  # counterpart for url_for()
    if (LOCAL):
        engine = create_engine("postgresql://postgres:Michigan123@localhost:5433/forecast_ai")
    else:
        engine = create_engine(
            'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai')
    query = "select series_id, min, max, mean, sum from stats where username = '{}'".format(session['username'])
    df = pd.read_sql(query, engine)
    results = [['ID','Minimum','Maximum','Average','Sum']]
    for i in range(len(df)):
        results.append(df.values[i])
    return render_template('workflow.html', table=results) # Application

@socketio.on('connection_msg')
def connected(message):
    data = message
    print(data)

@socketio.on('forecast_settings')
def forecast_settings(message):
    if (LOCAL):
        engine = create_engine("postgresql://postgres:Michigan123@localhost:5433/forecast_ai")
    else:
        engine = create_engine(
            'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai')
    query = "select series_id, date, value from data where username = '{}'".format(session['username'])
    original_dataset = pd.read_sql(query, engine)
    original_dataset.columns = ['id','date','demand']
    original_dataset['date'] = pd.to_datetime(original_dataset['date'])

    column_headers = preprocessing(original_dataset)

    # Set the time unit and metrc unit names
    time_unit = column_headers[0]
    metric_unit = column_headers[1]
    id_unit = 'id'

    # Determine whether the timeframe is daily, weekly, monthly, or yearly
    timeframe = determine_timeframe(original_dataset, time_unit)

    # Initial forecast settings - the first time the user sends forecast settings through the app - will use this value in forecastr method
    build_settings = 'initial'

    # store message['data'] into a df called data
    data = message['data']


    #print("******************** ORIGINAL DATASET *****************************")
    time_series_data = pd.DataFrame(original_dataset)
    time_series_data = time_series_data[time_series_data['id'] == time_series_data['id'].values[0]]
    forecast_settings = data[0]
    freq = data[2]
    print("Column Headers",column_headers)

    # Format the date and metric unit
    time_unit = column_headers[0]
    time_series_data[time_unit] = pd.to_datetime(time_series_data[time_unit])
    metric = column_headers[1]

    forecast = multiForecast(pd.DataFrame(original_dataset),forecast_settings,column_headers,freq,build_settings)

    # Need to convert forecast back into a list / array for y, y_hat and date so it can be properly graphed with chartjs
    y_hat = forecast[0]
    y = forecast[6]
    dates = forecast[1]
    model = forecast[2]
    csv_export = forecast[3]
    forecasted_vals = forecast[4]
    forecasted_vals_mean = forecast[5]
    validation_score = forecast[7]
    benchmark_validation_score = forecast[8]
    #original_dataset['date'] = original_dataset['date'].apply(lambda x: str(x).split(' ')[0])
    original_dataset = original_dataset.to_json()
    # Send data back to the client
    data_back_to_client = [dates,y_hat,y,forecast_settings,column_headers,freq, original_dataset, csv_export, forecasted_vals, forecasted_vals_mean, validation_score,
                           benchmark_validation_score]

    print("Complete")
    emit('render_forecast_chart', {'data': data_back_to_client})


@socketio.on('reset')
def reset(message):
    data = message['data']
    #print(data)

@socketio.on('update_data')
def main(message):
    print("Updating Postgres database")
    # Store message['data'] in data
    data = message['data']

    if(LOCAL):
        conn = psycopg2.connect(
            "dbname=forecast_ai user=postgres password=Michigan123 port=5433")
    else:
        conn = psycopg2.connect("dbname=forecast_ai user=postgres password=Michigan123 host=valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com port=5432")
    # Convert data to a pandas DataFrame
    data = pd.DataFrame(data)
    if (LOCAL):
        engine = create_engine("postgresql://postgres:Michigan123@localhost:5433/forecast_ai")
    else:
        engine = create_engine(
            'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai')

    df = data.copy()
    df['username'] = session['username']
    df.columns = ['series_id','date','value','username']
    df = df[['username','series_id','date','value']]
    cur = conn.cursor()
    print("Deleting from data")
    query = "DELETE FROM data where username = '{}'".format(session['username'])
    cur.execute(query)
    conn.commit()

    print("Deleting from stats")
    query = "DELETE FROM stats where username = '{}'".format(session['username'])
    cur.execute(query)
    conn.commit()
    print("Writing to data")
    df.head(0).to_sql('data', engine, if_exists='append', index=False)  # truncates the table

    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()

    df = df.set_index(df.index.astype(str) + '_' + session['username'])
    df.to_csv(output, sep='\t', header=False, index=True)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, 'data', null="")  # null values become ''
    conn.commit()
    #df.to_sql('data', engine, if_exists='append', index=False)

    aggcols = {
        'value':['min','max','mean','sum']
    }
    df_stats = df.groupby(['series_id','username']).agg(aggcols)
    df_stats.columns = ['min','max','mean','sum']
    df_stats.reset_index(inplace=True)
    print("Writing to stats")
    df_stats.to_sql('stats', engine, if_exists='append', index=False)

    cur.close()
    conn.close()
    print("Update Completed")

    column_headers = preprocessing(data)

    # Set the time unit and metrc unit names
    time_unit = column_headers[0]
    metric_unit = column_headers[1]
    id_unit = 'id'

    # Determine whether the timeframe is daily, weekly, monthly, or yearly
    timeframe = determine_timeframe(data, time_unit)

    agg = data.groupby(['date'], as_index=False)['demand'].sum()

    # Get summary statistics about original dataset
    summary_stats = get_summary_stats(agg, column_headers)
    count = len(data['id'].unique())
    summary_stats.append(count)

    agg_dimension = agg[time_unit].astype(str).tolist()
    agg_metric = agg[metric_unit].tolist()

    # Send original data to a list
    id = data[id_unit].tolist()
    dimension = data[time_unit].tolist()
    metric = data[metric_unit].tolist()

    data['date'] = pd.to_datetime(data['date'])

    emit('refresh_overview_table', {'data': [column_headers]})
    return render_template('login.html')

@socketio.on('build_model')
def model(message):
    if (LOCAL):
        engine = create_engine("postgresql://postgres:Michigan123@localhost:5433/forecast_ai")
    else:
        engine = create_engine(
            'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai')
    query = "select series_id, date, value from data where username = '{}'".format(session['username'])
    data = pd.read_sql(query, engine)
    data.columns = ['id', 'date', 'demand']
    data['date'] = pd.to_datetime(data['date'])

    # Let's do some preprocessing on this data to determine which column is the dimension vs. metric.
    column_headers = preprocessing(data)

    # Set the time unit and metrc unit names
    time_unit = column_headers[0]

    # Determine whether the timeframe is daily, weekly, monthly, or yearly
    timeframe = determine_timeframe(data, time_unit)

    data['date'] = pd.to_datetime(data['date'])

    # Get summary statistics about original dataset
    summary_stats = get_summary_stats(data, column_headers)
    count = len(data['id'].unique())
    summary_stats.append(count)

    # Send data back to the client in the form of a label detected or text extracted.
    emit('build_model', {'data': [column_headers,message, timeframe, summary_stats]})


if __name__ == '__main__':
    #app.run(port=5000)
    socketio.run(app, log_output=False)