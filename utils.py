import numpy as np
import pandas as pd

def rmse(ytrue, ypred):
    return np.round(np.sqrt(np.mean((np.nan_to_num(ytrue) - np.nan_to_num(ypred))**2)),2)

def mae(ytrue, ypred):
    return np.round(np.mean(np.abs(np.nan_to_num(ytrue) - np.nan_to_num(ypred))),2)

def mape(ytrue, ypred):
    mask = ytrue != 0
    return np.round(np.mean((np.abs(ytrue[mask] - ypred[mask]) / ytrue[mask])), 2)

def financial(ytrue, ypred, over, under):
    res = ytrue - ypred
    res_pos = res[res > 0]
    res_neg = np.abs(res[res <=0])
    res_pos = res_pos * over
    res_neg = res_neg * under
    val = np.sum(res_pos) + np.sum(res_neg)
    return np.round(val,0)

def get_timespan(df, dt, minus, periods):
    cols = ['d{}'.format(c) for c in range(dt + minus - 1, dt - 1, -1)]
    return df[cols]


def prepare_dataset_weekly(df, t, fclen, is_train=True, name_prefix=None):
    X = {"sales_4": get_timespan(df, t, 4, 4).sum(axis=1).values,
         "sales_16": get_timespan(df, t, 16, 16).sum(axis=1).values,
         "sales_52": get_timespan(df, t, 52, 52).sum(axis=1).values}

    for i in [3, 8, 16, 26, 52]:
        tmp = get_timespan(df, t, i, i)
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values

    for i in [3, 8, 16, 26, 52]:
        tmp = get_timespan(df, t + 1, i, i)
        X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s_2' % i] = tmp.mean(axis=1).values
        X['median_%s_2' % i] = tmp.median(axis=1).values
        X['min_%s_2' % i] = tmp.min(axis=1).values
        X['max_%s_2' % i] = tmp.max(axis=1).values
        X['std_%s_2' % i] = tmp.std(axis=1).values

    for i in range(fclen):
        X['lag_%s' % i] = get_timespan(df, t + i, 1, 1).values.ravel()

    X = pd.DataFrame(X)

    if is_train:
        cols = ['d{}'.format(c) for c in range(t - 1, t - fclen - 1, -1)]
        y = df[cols].values
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X


def prepare_dataset_monthly(df, t, fclen, is_train=True, name_prefix=None):
    X = {"sales_4": get_timespan(df, t, 4, 4).sum(axis=1).values,
         "sales_12": get_timespan(df, t, 16, 16).sum(axis=1).values}

    for i in [3, 6, 9, 12]:
        tmp = get_timespan(df, t, i, i)
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values

    for i in [3, 6, 9, 12]:
        tmp = get_timespan(df, t + 1, i, i)
        X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s_2' % i] = tmp.mean(axis=1).values
        X['median_%s_2' % i] = tmp.median(axis=1).values
        X['min_%s_2' % i] = tmp.min(axis=1).values
        X['max_%s_2' % i] = tmp.max(axis=1).values
        X['std_%s_2' % i] = tmp.std(axis=1).values

    for i in range(fclen):
        X['lag_%s' % i] = get_timespan(df, t + i, 1, 1).values.ravel()

    X = pd.DataFrame(X)

    if is_train:
        cols = ['d{}'.format(c) for c in range(t - 1, t - fclen - 1, -1)]
        y = df[cols].values
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X

def prepare_dataset_daily(df, t, fclen, is_train=True, name_prefix=None):
    X = {"sales_7": get_timespan(df, t, 7, 7).sum(axis=1).values,
         "sales_14": get_timespan(df, t, 14, 14).sum(axis=1).values,
         "sales_28": get_timespan(df, t, 28, 28).sum(axis=1).values,
         "sales_60": get_timespan(df, t, 60, 60).sum(axis=1).values}

    for i in [3, 7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t, i, i)
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values

    for i in [3, 7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t + 7, i, i)
        X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s_2' % i] = tmp.mean(axis=1).values
        X['median_%s_2' % i] = tmp.median(axis=1).values
        X['min_%s_2' % i] = tmp.min(axis=1).values
        X['max_%s_2' % i] = tmp.max(axis=1).values
        X['std_%s_2' % i] = tmp.std(axis=1).values

    for i in [7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t, i, i)
        X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
        X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

    for i in range(1, fclen):
        X['day_%s' % i] = get_timespan(df, t + 1, 1, 1).values.ravel()

    X = pd.DataFrame(X)

    if is_train:
        cols = ['d{}'.format(c) for c in range(t - 1, t - fclen - 1, -1)]
        y = df[cols].values
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X

def prepare_for_daily_model(df, fclen):
    num_days = min(df.shape[1] - 4 - 147 - fclen, 50)
    num_days = 40

    t = 2*fclen + num_days
    X_l, y_l = [], []

    for i in range(num_days):
        delta = i
        X_tmp, y_tmp = prepare_dataset_daily(df, t - delta, fclen=fclen)

        X_l.append(X_tmp)
        y_l.append(y_tmp)

    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)

    X_val, y_val = prepare_dataset_daily(df, fclen + 1, fclen=fclen)
    X_test = prepare_dataset_daily(df, 1, is_train=False, fclen=fclen)

    del X_l, y_l

    return X_train, X_val, X_test, y_train, y_val

def prepare_for_weekly_model(df, fclen):
    num_days = min(df.shape[1] - 4 - 147 - fclen, 50)
    num_days = 30

    t = 2 * fclen + num_days
    X_l, y_l = [], []

    for i in range(num_days):
        delta = i
        X_tmp, y_tmp = prepare_dataset_weekly(df, t - delta, fclen=fclen)

        X_l.append(X_tmp)
        y_l.append(y_tmp)

    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)

    X_val, y_val = prepare_dataset_weekly(df, fclen + 1, fclen=fclen)
    X_test = prepare_dataset_weekly(df, 1, fclen=fclen, is_train=False)

    del X_l, y_l

    return X_train, X_val, X_test, y_train, y_val