import pandas as pd
import numpy as np
import lightgbm as lgb
import keras as ks
from datetime import date, timedelta
from utils import rmse, mae, mape, financial, prepare_dataset_daily, prepare_dataset_monthly, prepare_dataset_weekly,\
    prepare_for_daily_model, prepare_for_weekly_model, get_timespan
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from sklearn.preprocessing import StandardScaler

class StepwiseLightGBM():
    def __init__(self, demandTable, fs_period, frequency, metric, over, under):
        tmp = pd.pivot_table(demandTable, values='demand', columns='date', index='id').fillna(0).reset_index()
        tmp = pd.melt(tmp, id_vars='id', var_name='date', value_name='demand', value_vars=tmp.drop(['id'], axis=1).columns.tolist())
        self.demandTable = tmp.sort_values('date', ascending=True).reset_index(drop=True)
        self.columns = demandTable.columns.tolist()
        self.fclen = fs_period
        self.frequency = frequency
        self.metric = metric
        self.over = over
        self.under = under
        self.maxDate = pd.to_datetime(demandTable[demandTable.columns[1]]).max()
        timeskip = 1
        if(frequency=='weekly'):
            timeskip = 7
        self.validation_cutoff = self.maxDate - timedelta(self.fclen * timeskip)

    def prepare_data(self):
        id = self.columns[0]  # id
        date = self.columns[1]  # date
        value = self.columns[2]  # value

        df = pd.pivot_table(self.demandTable, values=value, index=id, columns=date).fillna(0)
        df.columns = ['d{}'.format(c) for c in range(df.shape[1], 0, - 1)]
        df.reset_index(inplace=True)
        return df

    def run_lgb(self, X_train, X_val, X_test, y_train, y_val, uid):
        if (self.metric == 'rmse'):
            params = {
                'num_leaves': 80,
                'objective': 'regression',
                'min_data_in_leaf': 20,
                'learning_rate': 0.03,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.7,
                'bagging_freq': 1,
                'metric': 'rmse',
                'num_threads': 16,
                'verbose': 0
            }
        else:
            params = {
                'num_leaves': 80,
                'objective': 'regression',
                'min_data_in_leaf': 20,
                'learning_rate': 0.03,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.7,
                'bagging_freq': 1,
                'metric': 'mae',
                'num_threads': 16,
                'verbose': 0
            }

        MAX_ROUNDS = 5000
        train_pred = []
        val_pred = []
        test_pred = []
        cate_vars = []
        feat_imps = []
        for i in range(self.fclen):
            print("=" * 50)
            print("Step %d" % (i + 1))
            print("=" * 50)
            dtrain = lgb.Dataset(
                X_train, label=y_train[:, i],
                categorical_feature=cate_vars
            )
            dval = lgb.Dataset(
                X_val, label=y_val[:, i], reference=dtrain,
                categorical_feature=cate_vars)
            bst = lgb.train(
                params, dtrain, num_boost_round=MAX_ROUNDS,
                valid_sets=[dtrain, dval], early_stopping_rounds=125, verbose_eval=50
            )
            feat_importance = pd.DataFrame(
                {'Feature': X_train.columns, 'Importance': bst.feature_importance("gain")})

            val_pred.append(bst.predict(
                X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
            test_pred.append(bst.predict(
                X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))
            train_pred.append(bst.predict(
                X_train, num_iteration=bst.best_iteration or MAX_ROUNDS))
            feat_imps.append(feat_importance.sort_values('Importance', ascending=False).reset_index(drop=True))

        df_val = pd.DataFrame(
            np.array(val_pred).transpose(),
            columns=[f'F{i}' for i in range(1, self.fclen + 1)]
        )
        df_preds = pd.DataFrame(
            np.array(test_pred).transpose(),
            columns=[f'F{i}' for i in range(1, self.fclen + 1)]
        )
        df_train = pd.DataFrame(
            np.array(train_pred).transpose(),
            columns=[f'F{i}' for i in range(1, self.fclen + 1)]
        )

        predcols = df_val.columns[-self.fclen:]
        df_true = pd.DataFrame(y_val, columns=predcols)
        df_val.columns = [str(self.maxDate - timedelta(self.fclen-1) + timedelta(i)).split(' ')[0] for i in range(len(df_val.columns))]
        df_preds.columns = [str(self.maxDate + timedelta(i+1)).split(' ')[0] for i in range(len(df_preds.columns))]
        df_train.columns = [str(self.maxDate - timedelta(len(df_train.columns)) + timedelta(i)).split(' ')[0] for i in range(len(df_train.columns))]
        df_true['id'] = uid
        df_train['id'] = uid
        df_val['id'] = uid
        df_preds['id'] = uid
        df_val = pd.melt(df_val, id_vars='id', value_vars=df_val.drop(['id'], axis=1).columns.tolist(),
                         value_name='demand', var_name='date')
        df_preds = pd.melt(df_preds, id_vars='id', value_vars=df_preds.drop(['id'], axis=1).columns.tolist(),
                           value_name='forecasted value', var_name='date')
        df_true = pd.melt(df_true, id_vars='id', value_vars=df_true.drop(['id'], axis=1).columns.tolist(),
                          value_name='demand', var_name='date')
        df_train = pd.melt(df_train, id_vars='id', value_vars=df_train.drop(['id'], axis=1).columns.tolist(),
                          value_name='demand', var_name='date')

        del bst, val_pred, test_pred, X_train, X_val, X_test, dtrain, dval

        return df_preds, df_val, df_true, df_train, feat_imps

    def get_forecast(self):
        print("Beginning LightGBM")
        df = self.prepare_data()
        print(df.head())
        print(df.shape)
        print("Data Prepared")
        if (self.frequency == 'daily'):
            X_train, X_val, X_test, y_train, y_val = prepare_for_daily_model(df, fclen=self.fclen)
        elif (self.frequency == 'weekly'):
            X_train, X_val, X_test, y_train, y_val = prepare_for_weekly_model(df, fclen=self.fclen)
        print("Train set built")
        df_preds, df_val, df_true, df_train, feat_imps = self.run_lgb(X_train, X_val, X_test, y_train, y_val, df['id'])
        cutoff = len(df[df.columns[0]].unique()) * self.fclen
        traindates = 365 * len(df[df.columns[0]].unique())
        trainpreds = len(df_train['date'].unique()) * len(df[df.columns[0]].unique())
        #res = self.demandTable[-traindates:-cutoff]
        #val = self.demandTable[-cutoff:]
        res = self.demandTable[self.demandTable.date <= self.validation_cutoff].reset_index(drop=True)
        val = self.demandTable[self.demandTable.date > self.validation_cutoff].reset_index(drop=True)
        res['time'] = "Train"
        val['time'] = 'Validation'
        val['date'] = val['date'].astype(str)
        df_val['date'] = df_val['date'].astype(str)
        tmp = val[['id','date']].merge(df_val[['id','date','demand']], on=['id','date'], how='left')
        val['forecasted value'] = tmp['demand'].values
        metric_table = val.copy()
        metric_table['residual'] = metric_table['demand'] - metric_table['forecasted value']
        metric_table['mae'] = np.abs(metric_table['forecasted value'] - metric_table['demand'])
        metric_table['mse'] = (metric_table['forecasted value'] - metric_table['demand'])**2
        metric_table = metric_table[['id','date','residual','mae','mse']]
        MAE = mae(val['demand'], val['forecasted value'])
        RMSE = rmse(val['demand'], val['forecasted value'])
        MAPE = mape(val['demand'], val['forecasted value'])
        CASH = financial(val['demand'], val['forecasted value'], over=self.over, under=self.under)
        metrics = [RMSE, MAE, CASH]
        #tmp[-trainpreds:] = df_train['demand'].values
        #res['forecasted value'] = res['demand'].values
        res['forecasted value'] = np.nan
        #res['date'] = df_train['date']
        df_preds['time'] = 'Forecast'
        df_preds['demand'] = np.nan
        res = pd.concat([res, val, df_preds], axis=0).reset_index(drop=True)
        res['date']= pd.to_datetime(res['date'])

        last_period = val.groupby(['id'], as_index=False)['demand'].sum()
        next_period = df_preds.groupby(['id'], as_index=False)['forecasted value'].sum()
        change = last_period.merge(next_period, how='left', on=['id'])
        change.columns = ['series_id','last_total','next_total']
        change['change'] = np.round(100*(change['next_total'] - change['last_total']) / change['last_total'], 2)
        change['change'] = change['change'].astype(str) + '%'

        return res, df_preds, metrics, metric_table, change

class MLP():
    def __init__(self, demandTable, fs_period, frequency, metric, over, under):
        self.demandTable = demandTable
        self.columns = demandTable.columns.tolist()
        self.fclen = fs_period
        self.frequency = frequency
        self.metric = metric
        self.over = over
        self.under = under
        self.maxDate = pd.to_datetime(demandTable[demandTable.columns[1]]).max()
        timeskip = 1
        if (frequency == 'weekly'):
            timeskip = 7
        self.validation_cutoff = self.maxDate - timedelta(self.fclen * timeskip)

    def prepare_data(self):
        id = self.columns[0]  # id
        date = self.columns[1]  # date
        value = self.columns[2]  # value

        df = pd.pivot_table(self.demandTable, values=value, index=id, columns=date)
        df.columns = ['d{}'.format(c) for c in range(df.shape[1], 0, - 1)]
        df[:] = np.clip(df.values, 0, 1000000)
        #df[:] = np.log1p(df.values)
        df.reset_index(inplace=True)
        return df.fillna(0)

    def get_model(self, input_shape, output_shape, loss):
        inp = ks.layers.Input(shape=(input_shape,))

        x = ks.layers.Dense(200, activation='relu')(inp)
        x = ks.layers.Dropout(.15)(x)

        x = ks.layers.Dense(200, activation='relu')(x)
        x = ks.layers.Dropout(.15)(x)

        x = ks.layers.Dense(100, activation='relu')(x)

        out = ks.layers.Dense(output_shape, activation='linear')(x)
        model = ks.models.Model(input=inp, output=out)
        model.compile(optimizer='adam', loss=loss)
        return model


    def run_mlp(self, X_train, X_val, X_test, y_train, y_val, uid):
        print(self.validation_cutoff)
        if (self.metric == 'rmse'):
            model = self.get_model(input_shape=X_train.shape[1], output_shape=self.fclen, loss='mse')
        else:
            model = self.get_model(input_shape=X_train.shape[1], output_shape=self.fclen, loss='mae')

        callback = [ks.callbacks.EarlyStopping(patience=5),
                ks.callbacks.ReduceLROnPlateau(patience=3, verbose=2),
                ks.callbacks.ModelCheckpoint('mlp.hdf',save_best_only=True,save_weights_only=True)]

        y_mean = np.mean(y_train, axis=0)
        history = model.fit(X_train, y_train - y_mean, batch_size=32, epochs=150, validation_data = (X_val, y_val - y_mean), verbose=2, callbacks=callback)
        model.load_weights('mlp.hdf')
        val_pred = model.predict(X_val) + y_mean
        test_pred = model.predict(X_test) + y_mean

        df_val = pd.DataFrame(val_pred,
            columns=[f'F{i}' for i in range(1, self.fclen + 1)]
        )
        df_preds = pd.DataFrame(test_pred,
            columns=[f'F{i}' for i in range(1, self.fclen + 1)]
        )

        predcols = df_val.columns[-self.fclen:]
        df_true = pd.DataFrame(y_val, columns=predcols)
        df_val.columns = [str(self.maxDate - timedelta(self.fclen) + timedelta(i)).split(' ')[0] for i in
                          range(len(df_val.columns))]
        df_preds.columns = [str(self.maxDate + timedelta(i+1)).split(' ')[0] for i in range(len(df_preds.columns))]
        df_true['id'] = uid
        df_val['id'] = uid
        df_preds['id'] = uid
        df_val = pd.melt(df_val, id_vars='id', value_vars=df_val.drop(['id'], axis=1).columns.tolist(),
                         value_name='demand', var_name='date')
        df_preds = pd.melt(df_preds, id_vars='id', value_vars=df_preds.drop(['id'], axis=1).columns.tolist(),
                           value_name='forecasted value', var_name='date')
        df_true = pd.melt(df_true, id_vars='id', value_vars=df_true.drop(['id'], axis=1).columns.tolist(),
                          value_name='demand', var_name='date')

        del val_pred, test_pred, X_train, X_val, X_test

        return df_preds, df_val, df_true

    def get_forecast(self):
        print("Beginning MLP")
        df = self.prepare_data()
        if (self.frequency == 'daily'):
            X_train, X_val, X_test, y_train, y_val = prepare_for_daily_model(df, fclen=self.fclen)
        elif (self.frequency == 'weekly'):
            X_train, X_val, X_test, y_train, y_val = prepare_for_weekly_model(df, fclen=self.fclen)

        sc = StandardScaler()
        X_train[:] = np.nan_to_num(sc.fit_transform(X_train.fillna(0)))
        X_val[:] = np.nan_to_num(sc.transform(X_val.fillna(0)))
        X_test[:] = np.nan_to_num(sc.transform(X_test.fillna(0)))
        print("Train set built")
        df_preds, df_val, df_true = self.run_mlp(X_train, X_val, X_test, y_train, y_val,
                                                                      df['id'])
        cutoff = len(df[df.columns[0]].unique()) * self.fclen
        traindates = 365 * len(df[df.columns[0]].unique())
        res = self.demandTable[self.demandTable.date <= self.validation_cutoff].reset_index(drop=True)
        val = self.demandTable[self.demandTable.date > self.validation_cutoff].reset_index(drop=True)
        res['time'] = "Train"
        val['time'] = 'Validation'
        val['forecasted value'] = df_val['demand'].values
        metric_table = val.copy()
        metric_table['residual'] = metric_table['demand'] - metric_table['forecasted value']
        metric_table['mae'] = np.abs(metric_table['forecasted value'] - metric_table['demand'])
        metric_table['mse'] = (metric_table['forecasted value'] - metric_table['demand'])**2
        metric_table = metric_table[['id','date','residual','mae','mse']]

        MAE = mae(val['demand'], val['forecasted value'])
        RMSE = rmse(val['demand'], val['forecasted value'])
        CASH = financial(val['demand'], val['forecasted value'], over=self.over, under=self.under)
        metrics = [RMSE, MAE, CASH]
        # tmp[-trainpreds:] = df_train['demand'].values
        #res['forecasted value'] = res['demand'].values
        res['forecasted value'] = np.nan
        # res['date'] = df_train['date']
        df_preds['time'] = 'Forecast'
        df_preds['demand'] = np.nan
        res = pd.concat([res, val, df_preds], axis=0).reset_index(drop=True)
        res['date'] = pd.to_datetime(res['date'])

        last_period = val.groupby(['id'], as_index=False)['demand'].sum()
        next_period = df_preds.groupby(['id'], as_index=False)['forecasted value'].sum()
        change = last_period.merge(next_period, how='left', on=['id'])
        change.columns = ['series_id','last_total','next_total']
        change['change'] = np.round(100*(change['next_total'] - change['last_total']) / change['last_total'], 2)
        change['change'] = change['change'].astype(str) + '%'
        return res, df_preds, metrics, metric_table, change

class ExponentialSmoothing():
    def __init__(self, demandTable, fs_period, over, under):
        self.demandTable = demandTable
        self.columns = demandTable.columns.tolist()
        self.fclen = fs_period
        self.frequency = 'daily'
        self.over = over
        self.under = under
        self.maxDate = pd.to_datetime(demandTable[demandTable.columns[1]]).max()

    def prepare_data(self):
        id = self.columns[0]  # id
        date = self.columns[1]  # date
        value = self.columns[2]  # value

        df = pd.pivot_table(self.demandTable, values=value, index=id, columns=date)
        df.columns = ['d{}'.format(c) for c in range(df.shape[1], 0, - 1)]
        df.reset_index(inplace=True)
        return df

    def run_ets(self, train, uid):
        oofs = np.zeros((len(train), self.fclen))
        preds = oofs.copy()
        print(train.columns)
        for i in range(len(train)):
            ts1 = pd.Series(train.drop(['id'], axis=1).values[i][:-self.fclen])
            ts2 = pd.Series(train.drop(['id'], axis=1).values[i])
            ses1 = SimpleExpSmoothing(ts1, initialization_method="heuristic").fit(smoothing_level=0.2,optimized=False)
            ses2 = SimpleExpSmoothing(ts2, initialization_method="heuristic").fit(smoothing_level=0.2, optimized=False)

            holt1 = Holt(ts1, initialization_method="estimated").fit().forecast(self.fclen)
            holt2 = Holt(ts2, initialization_method="estimated").fit().forecast(self.fclen)
            '''fit3 = Holt(ts1, exponential=True, initialization_method="estimated").fit()
            fcast3 = fit3.forecast(self.fclen).rename("Exponential")
            fit4 = Holt(ts1, damped_trend=True, initialization_method="estimated").fit(damping_trend=0.98)
            fcast4 = fit4.forecast(self.fclen).rename("Additive Damped")
            fit5 = Holt(ts1, exponential=True, damped_trend=True, initialization_method="estimated").fit()
            fcast5 = fit5.forecast(self.fclen).rename("Multiplicative Damped")'''
            fcast1 = ses1.forecast(self.fclen)
            fcast2 = ses2.forecast(self.fclen)
            oofs[i] = holt1
            preds[i] = holt2

        df_val = pd.DataFrame(oofs,
            columns=[f'F{i}' for i in range(1, self.fclen + 1)]
        )
        df_preds = pd.DataFrame(preds,
            columns=[f'F{i}' for i in range(1, self.fclen + 1)]
        )

        predcols = df_val.columns[-self.fclen:]
        df_true = pd.DataFrame(train.values[:,-self.fclen:], columns=predcols)
        df_val.columns = [str(self.maxDate - timedelta(self.fclen) + timedelta(i)).split(' ')[0] for i in range(len(df_val.columns))]
        df_preds.columns = [str(self.maxDate + timedelta(i+1)).split(' ')[0] for i in range(len(df_preds.columns))]
        df_true['id'] = uid
        df_val['id'] = uid
        df_preds['id'] = uid
        df_val = pd.melt(df_val, id_vars='id', value_vars=df_val.drop(['id'], axis=1).columns.tolist(),
                         value_name='demand', var_name='date')
        df_preds = pd.melt(df_preds, id_vars='id', value_vars=df_preds.drop(['id'], axis=1).columns.tolist(),
                           value_name='forecasted value', var_name='date')
        df_true = pd.melt(df_true, id_vars='id', value_vars=df_true.drop(['id'], axis=1).columns.tolist(),
                          value_name='demand', var_name='date')


        return df_preds, df_val, df_true


    def get_forecast(self):
        print("Beginning Exponential Smoothing")
        df = self.prepare_data()
        print("Data Prepared")

        print("Train set built")
        df_preds, df_val, df_true = self.run_ets(df, df['id'])
        cutoff = len(df[df.columns[0]].unique()) * self.fclen
        traindates = 365 * len(df[df.columns[0]].unique())
        res = self.demandTable[-traindates:-cutoff]
        val = self.demandTable[-cutoff:]
        res['time'] = "Train"
        val['time'] = 'Validation'
        val['forecasted value'] = df_val['demand'].values
        metric_table = val.copy()
        metric_table['residual'] = metric_table['demand'] - metric_table['forecasted value']
        metric_table['mae'] = np.abs(metric_table['forecasted value'] - metric_table['demand'])
        metric_table['mse'] = (metric_table['forecasted value'] - metric_table['demand'])**2
        metric_table = metric_table[['id','date','residual','mae','mse']]

        MAE = mae(val['demand'], val['forecasted value'])
        RMSE = rmse(val['demand'], val['forecasted value'])
        MAPE = mape(val['demand'], val['forecasted value'])
        CASH = financial(val['demand'], val['forecasted value'], self.over, self.under)
        metrics = [RMSE, MAE, CASH]
        #tmp[-trainpreds:] = df_train['demand'].values
        res['forecasted value'] = res['demand'].values
        #res['date'] = df_train['date']
        df_preds['time'] = 'Forecast'
        df_preds['demand'] = np.nan
        print(df_preds.head())
        res = pd.concat([res, val, df_preds], axis=0).reset_index(drop=True)
        res['date']= pd.to_datetime(res['date'])

        last_period = val.groupby(['id'], as_index=False)['demand'].sum()
        next_period = df_preds.groupby(['id'], as_index=False)['forecasted value'].sum()
        change = last_period.merge(next_period, how='left', on=['id'])
        change.columns = ['series_id','last_total','next_total']
        change['change'] = np.round(100*(change['next_total'] - change['last_total']) / change['last_total'], 2)
        change['change'] = change['change'].astype(str) + '%'
        return res, df_preds, metrics, metric_table, change


class Naive():
    def __init__(self, demandTable, fs_period, over, under):
        self.demandTable = demandTable
        self.columns = demandTable.columns.tolist()
        self.fclen = fs_period
        self.frequency = 'daily'
        self.over = over
        self.under = under
        self.maxDate = pd.to_datetime(demandTable[demandTable.columns[1]]).max()

    def prepare_data(self):
        id = self.columns[0]  # id
        date = self.columns[1]  # date
        value = self.columns[2]  # value

        df = pd.pivot_table(self.demandTable, values=value, index=id, columns=date)
        df.columns = ['d{}'.format(c) for c in range(df.shape[1], 0, - 1)]
        df.reset_index(inplace=True)
        return df

    def run_naive(self, train, uid):
        val = train.values[:,-self.fclen:]
        val_preds = train.values[:,-2*self.fclen:-self.fclen]
        test_preds = train.values[:,-self.fclen:]
        df_val = pd.DataFrame(val_preds,
            columns=[f'F{i}' for i in range(1, self.fclen + 1)]
        )
        df_preds = pd.DataFrame(test_preds,
            columns=[f'F{i}' for i in range(1, self.fclen + 1)]
        )
        df_val.columns = [str(self.maxDate - timedelta(self.fclen) + timedelta(i)).split(' ')[0] for i in range(len(df_val.columns))]
        df_preds.columns = [str(self.maxDate + timedelta(i+1)).split(' ')[0] for i in range(len(df_preds.columns))]
        df_val['id'] = uid
        df_preds['id'] = uid
        df_val = pd.melt(df_val, id_vars='id', value_vars=df_val.drop(['id'], axis=1).columns.tolist(),
                         value_name='demand', var_name='date')
        df_preds = pd.melt(df_preds, id_vars='id', value_vars=df_preds.drop(['id'], axis=1).columns.tolist(),
                           value_name='forecasted value', var_name='date')
        return df_val, df_preds

    def run_snaive(self, train, uid, period_length=365):
        val = train.values[:,-self.fclen:]
        val_preds = train.shift(period_length-self.fclen, axis=1).values[:,-2*self.fclen:-self.fclen]
        test_preds = train.shift(period_length-self.fclen, axis=1).values[:,-self.fclen:]
        df_val = pd.DataFrame(val_preds,
            columns=[f'F{i}' for i in range(1, self.fclen + 1)]
        )
        df_preds = pd.DataFrame(test_preds,
            columns=[f'F{i}' for i in range(1, self.fclen + 1)]
        )
        df_val.columns = [str(self.maxDate - timedelta(self.fclen) + timedelta(i)).split(' ')[0] for i in range(len(df_val.columns))]
        df_preds.columns = [str(self.maxDate + timedelta(i+1)).split(' ')[0] for i in range(len(df_preds.columns))]
        df_val['id'] = uid
        df_preds['id'] = uid
        df_val = pd.melt(df_val, id_vars='id', value_vars=df_val.drop(['id'], axis=1).columns.tolist(),
                         value_name='demand', var_name='date')
        df_preds = pd.melt(df_preds, id_vars='id', value_vars=df_preds.drop(['id'], axis=1).columns.tolist(),
                           value_name='forecasted value', var_name='date')
        return df_val, df_preds

    def get_forecast(self):
        print("Beginning Exponential Smoothing")
        df = self.prepare_data()
        print("Data Prepared")

        print("Train set built")
        df_val, df_preds = self.run_naive(df, df['id'])
        cutoff = len(df[df.columns[0]].unique()) * self.fclen
        traindates = 365 * len(df[df.columns[0]].unique())
        res = self.demandTable[-traindates:-cutoff]
        val = self.demandTable[-cutoff:]
        res['time'] = "Train"
        val['time'] = 'Validation'
        val['forecasted value'] = df_val['demand'].values
        metric_table = val.copy()
        metric_table['residual'] = metric_table['demand'] - metric_table['forecasted value']
        metric_table['mae'] = np.abs(metric_table['forecasted value'] - metric_table['demand'])
        metric_table['mse'] = (metric_table['forecasted value'] - metric_table['demand'])**2
        metric_table = metric_table[['id','date','residual','mae','mse']]

        MAE = mae(val['demand'], val['forecasted value'])
        RMSE = rmse(val['demand'], val['forecasted value'])
        MAPE = mape(val['demand'], val['forecasted value'])
        CASH = financial(val['demand'], val['forecasted value'], over=self.over, under=self.under)
        metrics = [RMSE, MAE, CASH]
        #tmp[-trainpreds:] = df_train['demand'].values
        res['forecasted value'] = res['demand'].values
        #res['date'] = df_train['date']
        df_preds['time'] = 'Forecast'
        df_preds['demand'] = np.nan
        res = pd.concat([res, val, df_preds], axis=0).reset_index(drop=True)
        res['date']= pd.to_datetime(res['date'])
        return res, df_preds, metrics, metric_table

class Ensemble():
    def __init__(self, demandTable, fs_period, frequency, metric, over, under):
        self.demandTable = demandTable
        self.columns = demandTable.columns.tolist()
        self.fclen = fs_period
        self.frequency = frequency
        self.over = over
        self.under = under
        self.metric = metric
        self.maxDate = pd.to_datetime(demandTable[demandTable.columns[1]]).max()

    def prepare_data(self):
        id = self.columns[0]  # id
        date = self.columns[1]  # date
        value = self.columns[2]  # value

        df = pd.pivot_table(self.demandTable, values=value, index=id, columns=date)
        df.columns = ['d{}'.format(c) for c in range(df.shape[1], 0, - 1)]
        df.reset_index(inplace=True)
        return df.fillna(0)

    def get_forecast(self):
        gbm = StepwiseLightGBM(self.demandTable, self.fclen, self.frequency, self.metric, self.over, self.under)
        es = ExponentialSmoothing(self.demandTable, self.fclen, self.over, self.under)
        nnet = MLP(self.demandTable, self.fclen, self.metric, self.frequency, self.over, self.under)
        df = self.prepare_data()
        print("Data Prepared")
        print(self.frequency)
        if (self.frequency == 'daily'):
            X_train, X_val, X_test, y_train, y_val = prepare_for_daily_model(df, fclen=self.fclen)
        elif (self.frequency == 'weekly'):
            X_train, X_val, X_test, y_train, y_val = prepare_for_weekly_model(df, fclen=self.fclen)
        preds1, val1, df_true, df_train, feat_imps = gbm.run_lgb(X_train, X_val, X_test, y_train, y_val, df['id'])
        sc = StandardScaler()
        X_train[:] = np.nan_to_num(sc.fit_transform(X_train.fillna(0)))
        X_val[:] = np.nan_to_num(sc.transform(X_val.fillna(0)))
        X_test = np.nan_to_num(sc.transform(X_test.fillna(0)))
        print("Train set built")
        preds2, val2, _ = nnet.run_mlp(X_train, X_val, X_test, y_train, y_val, df['id'])
        preds3, val3, _ = es.run_ets(df, df['id'])
        df_preds = preds1.copy()
        df_val = val1.copy()
        df_preds['forecasted value'] = (preds1['forecasted value'] + preds2['forecasted value'] + preds3['forecasted value'])/3
        df_val['demand'] = (val1['demand'] + val2['demand'] + val3['demand']) / 3

        cutoff = len(df[df.columns[0]].unique()) * self.fclen
        traindates = 365 * len(df[df.columns[0]].unique())
        res = self.demandTable[-traindates:-cutoff]
        val = self.demandTable[-cutoff:]
        res['time'] = "Train"
        val['time'] = 'Validation'
        val['forecasted value'] = df_val['demand'].values
        metric_table = val.copy()
        metric_table['residual'] = metric_table['demand'] - metric_table['forecasted value']
        metric_table['mae'] = np.abs(metric_table['forecasted value'] - metric_table['demand'])
        metric_table['mse'] = (metric_table['forecasted value'] - metric_table['demand'])**2
        metric_table = metric_table[['id','date','residual','mae','mse']]

        MAE = mae(val['demand'], val['forecasted value'])
        RMSE = rmse(val['demand'], val['forecasted value'])
        MAPE = mape(val['demand'], val['forecasted value'])
        CASH = financial(val['demand'], val['forecasted value'], over=self.over, under=self.under)
        metrics = [RMSE, MAE, CASH]
        # tmp[-trainpreds:] = df_train['demand'].values
        res['forecasted value'] = res['demand'].values
        # res['date'] = df_train['date']
        df_preds['time'] = 'Forecast'
        df_preds['demand'] = np.nan
        res = pd.concat([res, val, df_preds], axis=0).reset_index(drop=True)
        res['date'] = pd.to_datetime(res['date'])

        last_period = val.groupby(['id'], as_index=False)['demand'].sum()
        next_period = df_preds.groupby(['id'], as_index=False)['forecasted value'].sum()
        change = last_period.merge(next_period, how='left', on=['id'])
        change.columns = ['series_id','last_total','next_total']
        change['change'] = np.round(100*(change['next_total'] - change['last_total']) / change['last_total'], 2)
        change['change'] = change['change'].astype(str) + '%'
        
        return res, df_preds, metrics, metric_table, change
