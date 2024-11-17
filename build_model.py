import pandas as pd
import numpy as np

import datetime
from dateutil.relativedelta import relativedelta

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm


def classic_model(train, test, type_of_aggregation, target_name):
    #1) prepearing data
    #2) build trend
    #3) calculate seasonal component
    #4) adjust trend
    #5) calculate metrics
### create data    
    # create data index for next period
    last_date = test.index[-1]
    index_add = []
    if type_of_aggregation == 'M':
        for i in range(1, 13):
            next_period = test.index[-1] + relativedelta(months=i)
            index_add.append(next_period)
    elif type_of_aggregation == 'W':
        for i in range(1, 40):
            next_period = test.index[-1] + relativedelta(weeks=i)
            index_add.append(next_period)
    elif type_of_aggregation == 'D':
        for i in range(1, 180):
            next_period = test.index[-1] + relativedelta(days=i)
            index_add.append(next_period)
            
    # to dataframe
    new_period_prediction = pd.DataFrame(index = index_add, columns = test.columns)
    
    # add new period as test
    test = pd.concat([test, new_period_prediction])

    # create time feature
    if train.shape[0] % 2 != 0:
        start = -int(train.shape[0] / 2)
        finish = start + train.shape[0]
        time = [i for i in range(start, finish)]
    
        test_start = finish
    else:
        start = (train.shape[0]) - 1
        finish = train.shape[0]
        time = [i for i in range(-start, finish, 2)]
    
        test_start = finish + 1

    if test.shape[0] % 2 != 0:
        finish = test_start + test.shape[0]
        test_time = [i for i in range(test_start, finish)]
    else:
        finish = train.shape[0]
        test_time = [i for i in range(-test_start, finish, 2)]
        
    # time feture for train
    train['time'] = 0
    train.loc[:, 'time'] = time

    # time feture for test
    test['time'] = 0
    test.loc[:, 'time'] = test_time

    full = pd.concat([train, test])

### build_model
    # Training data
    X = train.loc[:, ['time']]  # features
    y = train.loc[:, target_name]  # target
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Store the fitted values as a time series with the same time index as
    # the training data
    train['trend_prediction'] = pd.Series(model.predict(X), index=X.index)
    # write prediction
    test['trend_prediction'] = pd.Series(model.predict(test[['time']]), index=test.index)
    full['trend_prediction'] = pd.Series(model.predict(full[['time']]), index=full.index)
    
    # get metrics for linear model for trend
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    display(est2.summary())

    # plor residuals
    train['residuals'] = train[target_name] - train['trend_prediction']
    full['residuals'] = full[target_name] - full['trend_prediction']

    # calculate seasonal component
    temp = full.copy()

    temp['diff'] = temp[target_name] / temp['trend_prediction']
    temp['diff_shift'] = temp['diff'].shift(12)
    temp['seasonal'] = (temp['diff'] + temp['diff_shift']) / 2

    # to dataframe
    season = pd.DataFrame(temp.iloc[12:24, 7])
    
    # prolong seasonal component on the period of prediction
    season_long = pd.concat([season, season, season])
    
    # create right index
    first_date = train.index[0]
    end = season_long.shape[0]
    index_seasonal = []
    if type_of_aggregation == 'M':
        for i in range(end):
            next_period = first_date + relativedelta(months=i)
            index_seasonal.append(next_period)
    elif type_of_aggregation == 'W':
        for i in range(end):
            next_period = first_date + relativedelta(weeks=i)
            index_seasonal.append(next_period)
    elif type_of_aggregation == 'D':
        for i in range(end):
            next_period = first_date + relativedelta(days=i)
            index_seasonal.append(next_period)
    
    season_long.index = index_seasonal
    
    # add seasonal component to dataset
    test = test.merge(season_long, left_index = True, right_index = True, how = 'left')
    train = train.merge(season_long, left_index = True, right_index = True, how = 'left')
    
    # seasonal adjustment
    test['predict'] = test['trend_prediction'] * test['seasonal']
    train['predict'] = train['trend_prediction'] * train['seasonal']
    
    #check metrics
    real = train[target_name].copy()
    pred = train['predict'].copy()
    metrics_for_comparison = pd.DataFrame(index=['MSE', 'MAE', 'RMSE', 'R2'])
    
    metrics_for_comparison['Classic time series'] = [mean_squared_error(real, pred),
                                            mean_absolute_error(real, pred),
                                            np.sqrt(mean_squared_error(real, pred)),
                                            r2_score(real, pred)]
    print('\nMetrics after seasonal adjustment')
    display(metrics_for_comparison.round(decimals=3))
    
    # calculate results
    print('\nRESULT\n')
    display(f'The average monthly revenue ({target_name}) for the СLion for the first half of 2021 will be: ', round(test.iloc[3:9]['predict'].mean()))
    display(f'The product revenue ({target_name}) in the first half of 2021 will be:: ', round(test.iloc[3:9]['predict'].sum()))

    return train, test






