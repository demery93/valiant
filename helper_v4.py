import pandas as pd
import numpy as np
from flask_socketio import SocketIO, emit
from models import StepwiseLightGBM, Naive, ExponentialSmoothing, MLP, Ensemble
import time
from sqlalchemy import create_engine
import psycopg2
from flask import session
import io

def multiForecast(data,forecast_settings,column_headers,freq_val,build_settings):
    LOCAL=False
    # Hyper-parameters
    #fs_metric = forecast_settings[9]
    fs_metric = 'rmse'                                      # Setting everything to rmse
    #fs_model_type = forecast_settings[0]                    # linear or logistic
    fs_model_type = 'auto'                                  # Setting everything to auto
    fs_period = int(forecast_settings[0])                   # int
    over_penalty = int(forecast_settings[8])               # Cash penalty for over predicting
    under_penalty = int(forecast_settings[9])              # Cash penalty for under predicting
    frequency = forecast_settings[10]                       # Time series frequency

    # Initial Variables
    build = build_settings                                  # Determine the build_setting - either initial or update forecast settings.
    dimension = column_headers[0]                           # date
    metric = column_headers[1]                              # metric name

    print("Running Multi")
    model = StepwiseLightGBM(data, fs_period, frequency, fs_metric, over_penalty,under_penalty)
    '''if(fs_model_type=='auto'):
        model = Ensemble(data, fs_period, frequency, fs_metric, over_penalty, under_penalty)
    elif(fs_model_type=='lightgbm'):
        model = StepwiseLightGBM(data, fs_period, frequency, fs_metric, over_penalty,under_penalty)
    elif(fs_model_type=='ets'):
        model = ExponentialSmoothing(data, fs_period, over_penalty, under_penalty)
    elif(fs_model_type=='neural-net'):
        model = MLP(data, fs_period, frequency, fs_metric, over_penalty, under_penalty)'''
    model_naive = Naive(data, fs_period, over_penalty, under_penalty)
    forecast, df_preds, metrics, metrics_table, change_table = model.get_forecast()

    metrics_table.columns = ['series_id','date','residual','absolute_error','squared_error']
    metrics_table['username'] = session['username']
    forecast_naive, df_preds_naive, metrics_naive, naive_metrics_table = model_naive.get_forecast()

    metrics_table['benchmark_residual'] = naive_metrics_table['residual']

    ## Output forecast to postgres ##
    output = forecast.copy()
    #output = pd.concat([forecast, df_preds], axis=0).reset_index(drop=True).drop_duplicates()
    output['username'] = session['username']
    change_table['username'] = session['username']

    output.columns = ['series_id', 'date', 'value', 'time', 'forecast','username']
    output = output[['username','series_id', 'date', 'value', 'forecast']]

    if (LOCAL):
        engine = create_engine("postgresql://postgres:Michigan123@localhost:5433/forecast_ai")
        conn = psycopg2.connect("dbname=forecast_ai user=postgres password=Michigan123 port=5433")
    else:
        engine = create_engine(
            'postgresql://postgres:Michigan123@valiant.coujrcruw1hl.us-east-2.rds.amazonaws.com:5432/forecast_ai')

    conn = engine.raw_connection()
    cur = conn.cursor()
    query = "DELETE FROM output where username = '{}'".format(session['username'])
    cur.execute(query)
    conn.commit()

    query = "DELETE FROM metrics where username = '{}'".format(session['username'])
    cur.execute(query)
    conn.commit()

    query = "DELETE FROM standard_output where username = '{}'".format(session['username'])
    cur.execute(query)
    conn.commit()


    #metrics_table.to_sql('metrics', engine, if_exists='append', index=False)
    print("Writing metrics")
    metrics_table = metrics_table[['username','series_id','date','residual','absolute_error','squared_error','benchmark_residual']]
    metrics_table.head(0).to_sql('metrics', engine, if_exists='append', index=False)  # truncates the table
    conn = engine.raw_connection()
    cur = conn.cursor()
    output1 = io.StringIO()
    metrics_table.to_csv(output1, sep='\t', header=False, index=True)
    print("CSV written")
    output1.seek(0)
    contents = output1.getvalue()
    cur.copy_from(output1, 'metrics', null="")  # null values become ''
    conn.commit()
    #output.to_sql('output', engine, if_exists='append', index=False)
    print("Writing outputs")
    output.head(0).to_sql('output', engine, if_exists='append', index=False)  # truncates the table
    output = output[['username','series_id','date','value','forecast']]
    conn = engine.raw_connection()
    cur = conn.cursor()
    output1 = io.StringIO()
    output.to_csv(output1, sep='\t', header=False, index=True)
    print("CSV written")
    output1.seek(0)
    contents = output1.getvalue()
    cur.copy_from(output1, 'output', null="")  # null values become ''
    conn.commit()
    #change_table.to_sql('standard_output', engine, if_exists='append', index=False)
    print("Writing standard output")
    change_table.head(0).to_sql('standard_output', engine, if_exists='append', index=False)  # truncates the table
    change_table = change_table[['username','series_id','last_total','next_total','change']]

    conn = engine.raw_connection()
    cur = conn.cursor()
    output1 = io.StringIO()
    change_table.to_csv(output1, sep='\t', header=False, index=True)
    print("CSV written")
    output1.seek(0)
    contents = output1.getvalue()
    cur.copy_from(output1, 'standard_output', null="")  # null values become ''
    conn.commit()
    print("All tables exported to database")
    ## End output ##

    df_preds_agg = df_preds.groupby(['date'])['forecasted value'].sum().reset_index()

    df_agg = forecast.groupby(['date'])['forecasted value'].sum().reset_index()

    df_preds_agg.columns = ['date','yhat']
    df_agg.columns = ['date', 'yhat']
    y_hat = df_agg['yhat'].tolist()
    #dates = forecast['date'].unique().tolist() + df_preds_agg['date'].tolist()
    dates = df_agg['date'].astype(str).tolist()
    y = forecast.groupby(['date'])['demand'].sum().reset_index()['demand'].tolist()
    m = None
    csv_ready_for_export = forecast
    csv_ready_for_export.columns = ['id','date','y','time','yhat']
    csv_ready_for_export['yhat_upper'] = np.nan
    csv_ready_for_export['yhat_lower'] = np.nan
    csv_ready_for_export.drop(['time'], axis=1, inplace=True)

    csv_ready_for_export.columns = ['id','ds','y','yhat','yhat_upper','yhat_lower']
    # Select the columns we want to include in the export
    export_formatted = csv_ready_for_export[['id','ds','y','yhat','yhat_upper','yhat_lower']]
    export_formatted['ds'] = export_formatted['ds'].astype(str)

    # Rename y and yhat to the actual metric names
    export_formatted.rename(index=str, columns={'ds': 'date', 'y': metric, 'yhat': metric + '_forecast','yhat_upper':metric + '_upper_forecast','yhat_lower':metric + '_lower_forecast'}, inplace=True)
    export_formatted = export_formatted[['id','date',metric, metric+'_forecast']]
    # replace NaN with an empty val
    export_formatted = export_formatted.replace(np.nan, '', regex=True)

    # Format timestamp
    export_formatted['date'] = export_formatted['date'].apply(lambda x: str(x).split(' ')[0])
    # Create dictionary format for sending to csv
    csv_ready_for_export = export_formatted.to_dict('records')

    ##### Lets see how the forecast compares to historical performance #####

    # First, lets sum up the forecasted metric
    forecast_sum = df_preds_agg['yhat'][-fs_period:].sum()
    forecast_mean = df_preds_agg['yhat'][-fs_period:].mean()

    # Now lets sum up the actuals for the same time interval as we predicted
    agg = data.groupby(['date'])['demand'].sum().reset_index()
    actual_sum = float(agg['demand'][-fs_period:].sum())
    actual_mean = float(agg['demand'][-fs_period:].mean())

    difference = '{0:.1%}'.format(((forecast_sum - actual_sum) / forecast_sum))
    difference_mean = '{0:.1%}'.format(((forecast_mean - actual_mean) / forecast_mean))

    forecasted_vals = ['{0:.1f}'.format(forecast_sum),'{0:.1f}'.format(actual_sum),difference]
    forecasted_vals_mean = ['{0:.1f}'.format(forecast_mean),'{0:.1f}'.format(actual_mean),difference_mean]

    for i in range(len(y_hat[:-2*fs_period])):
        y_hat[i] = None
    #y_hat[:-2*fs_period] = ""
    y[-fs_period:] = ""
    #print(y_hat)

    return [y_hat, dates, m, csv_ready_for_export, forecasted_vals, forecasted_vals_mean, y, metrics, metrics_naive]


def get_summary_stats(data,column_headers):

    """

    Background:
    This function will get some summary statistics about the original dataset being uploaded.

    Input:

    data: a dataframe with the data from the uploaded csv containing a dimension and metric
    column_headers: string of column names for the dimension and metric


    Output:

    sum_stats: a list containing the count of time units, the mean, std, min and max values of the metric. This data is rendered on step 2 of the UI.

    """

    # Set the dimension and metrics
    dimension = column_headers[0]
    metric = column_headers[1]



    time_unit_count = str(data[dimension].count())





    print(data[metric].mean())

    mean = str(round(data[metric].mean(),2))
    print('string of the mean is ' + mean)


    std = str(round(data[metric].std(),2))
    minimum = str(round(data[metric].min(),2))
    maximum = str(round(data[metric].max(),2))

    sum_stats = [time_unit_count,mean,std,minimum,maximum]
    print(sum_stats)

    return sum_stats




def preprocessing(data):


    """

    Background: This function will determine which columns are dimensions (time_unit) vs metrics, in addition to reviewing the metric data to see if there are any objects in that column.

    Input:

        data (df): A dataframe of the parsed data that was uploaded.

    Output:

        [time_unit,metric_unit]: the appropriate column header names for the dataset.

    """

    # Get list of column headers
    column_headers = list(data)


    # Let's determine the column with a date


    col1 = column_headers[0]
    col2 = column_headers[1]
    col3 = column_headers[2]
    print('the second column is ' + col2)

    # Get the first value in column 1, which is what is going to be checked.
    col1_val = data[col1][0]
    print(type(col1_val))

    """

    TO DO: Pre-processing around the dtypes of both columns. If both are objects, I'll need to determine which is the column.

    TO DO: Emit any error messaging


    print('The data type of this metric column is: ' + str(data[metric].dtype))
    print(data[metric].head())

    data[metric] = data[metric].apply(lambda x: float(x))

    print(data[metric].dtype)


    """

    # Check to see if the data has any null values

    print('Is there any null values in this data? ' + str(data.isnull().values.any()))

    # If there is a null value in the dataset, locate it and emit the location of the null value back to the client, else continue:

    data = data[data[col1].notnull()].reset_index(drop=True) #drop nulls
    print(data.tail())

    do_nulls_exist = data.isnull().values.any()

    if do_nulls_exist == True:
        print('found a null value')
        #null_rows = pd.isnull(data).any(1).nonzero()[0]
        null_rows = (data.isnull().sum(axis=1) > 0).values.astype(int)
        print('######### ORIGINAL ROWS THAT NEED UPDATING ##############')
        print(null_rows)
        # Need to add 2 to each value in null_rows because there

        print('######### ROWS + 2 = ACTUAL ROW NUMBERS IN CSV ##############')
        update_these_rows = []
        for x in null_rows:

            update_these_rows.append(int(x)+2)

        print(update_these_rows)

        emit('error', {'data': update_these_rows})






    else:
        print('no nulls found')


    if isinstance(col1_val, (int, np.integer)) or isinstance(col1_val, float):
        print(str(col1_val) + ' this is a metric')
        print('Setting time_unit as the second column')
        time_unit = column_headers[1]
        metric_unit = column_headers[2]
        return [time_unit, metric_unit]
    else:
        print('Setting time_unit as the second column')
        time_unit = column_headers[1]
        metric_unit = column_headers[2]
        return [time_unit, metric_unit]






def determine_timeframe(data, time_unit):

    """

    Background:

    This function determines whether the data is daily, weekly, monthly or yearly by checking the delta between the first and second date in the df.

    Input:

    data: a df containg a dimension and a metric
    time_unit: is the dimension name for the date.


    Output:

    time_list: a list of strings to be used within the UI (time, desc) and when using the function future = m.make_future_dataframe(periods=fs_period, freq=freq_val)



    """


    # Determine whether the data is daily, weekly, monthly or yearly
    idcol = data.columns.tolist()[0]
    tmp = data[data[idcol] == data[idcol].unique()[0]].reset_index(drop=True)
    date1 = tmp[time_unit][0]
    date2 = tmp[time_unit][1]

    first_date = pd.Timestamp(tmp[time_unit][0])
    second_date = pd.Timestamp(tmp[time_unit][1])
    time_delta = second_date - first_date

    time_delta = int(str(time_delta).split(' ')[0])

    print([tmp[time_unit][0],tmp[time_unit][1]])
    print([second_date,first_date,time_delta])


    if time_delta == 1:
        time = 'days'
        freq = 'D'
        desc = 'daily'
    elif time_delta >=7 and time_delta <= 27:
        time = 'weeks'
        freq = 'W'
        desc = 'weekly'
    elif time_delta >=28 and time_delta <=31:
        time = 'months'
        freq = 'M'
        desc = 'monthly'
    elif time_delta >= 364:
        time = 'years'
        freq = 'Y'
        desc = 'yearly'
    else:
        print('error?')

    time_list = [time,freq, desc]
    #print(time_list)

    return time_list



    
