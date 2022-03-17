import pandas as pd
from pathlib import Path
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from scipy.fft import fft, ifft
### Lale dependencies
import lale
from lale.lib.lale import NoOp, Hyperopt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import StandardScaler as Standard
from xgboost import XGBRegressor as XGB
from lale.lib.sklearn import GradientBoostingRegressor as GradBoost, ExtraTreesRegressor, KNeighborsRegressor as KNN
from lale.lib.lightgbm import LGBMRegressor as LGBM
lale.wrap_imported_operators()


def read_file(fname):
    # the structure for oxygen, temp, and adcp are the same so
    # just read the file for each into df
    df = pd.read_csv(fname)
    return df

def split_data(X, y, train_size=0.8):
    ## split data along a given split

    split_loc = int(np.round(X.shape[0]*train_size)) # np.rint(X.shape[0]*train_size)
    X_train = X.iloc[0:split_loc, :]
    y_train = y.iloc[0:split_loc]
    X_test = X.iloc[split_loc:, :]
    y_test = y.iloc[split_loc: ]
    return X_train, X_test, y_train, y_test


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    subdirs = ['temperature/', 'oxygen/', 'adcp/']
    result_dir = './results/'
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    path_to_results = Path(result_dir)
    mae_results_file = path_to_results / 'ErrorMetrics_Lale.csv'
    f_results = open(mae_results_file, "w+")
    f_results.write("Sensor, MSE, MAE, RMSE\n")
    subd = subdirs[2] # We run on the ADCP directory
    for subd in subdirs:
        print('begin on subd', subd)
        Path(result_dir + subd).mkdir(parents=True, exist_ok=True)
        if subd != 'adcp/':
            continue
        print(subd)
        path_to_input_data = Path('../../sensor_data/timelagged_data') / subd # use path to be little more robust
        print(path_to_input_data)
        files = (glob.glob(str(path_to_input_data) + '/*.autoai'))

        for f in files[1:4]:
            print(f)
            df = read_file(f)
            df.drop(['week', 'dow', 'doq', 'qoy'], axis=1, inplace=True)
            # reverse because amlp sliding window is from the recent to the least recent
            df = df[::-1]
            print(df.keys())
            ## We want to act on the normalised data rather than complete signal
            ## hence let's take a diff over points 48
            nlag = 48
            df['output_res'] = df['output'].diff(periods=nlag)
            df['x1'] = df['x1'].diff(periods=nlag)
            df['x2'] = df['x2'].diff(periods=nlag)
            df['x3'] = df['x3'].diff(periods=nlag)
            df['x4'] = df['x4'].diff(periods=nlag)
            df['x5'] = df['x5'].diff(periods=nlag)
            df['x6'] = df['x6'].diff(periods=nlag)
            df = df.iloc[nlag:]
            X = df.drop(['output_res'], axis=1) # We are forecasting the residual
            y = df['output_res']
            X_train, X_test, y_train, y_test = split_data(X, y, train_size=0.8)
            ## We want to randomly shuffle X_train and y_train
            ## for convenience just use sklearn splitter
            ## we lose one time point but guess not important
            X_train, dum, y_train, dum = train_test_split(X_train,
                                                          y_train,train_size=0.999999999999999,random_state = 42) # train size of ~1 since we already split train test

            ## I don't think it makes sense to include month feature since we don't have multiyear data
            month_test_ = X_test['month'] #retain this for the output file
            year_test_ = X_test['year']  # retain this for the output file
            test_data_raw_ = X_test['output'] # This is the raw time series (not residuals); let's keep and write to the results file
            X_train.drop('month', axis=1, inplace = True)
            X_test.drop('month', axis=1, inplace = True)
            X_train.drop('year', axis=1, inplace=True)
            X_test.drop('year', axis=1, inplace=True)
            X_train.drop('output', axis=1, inplace=True)
            X_test.drop('output', axis=1, inplace=True)
            planned_pipe = (Standard | NoOp)  >> (  XGB | RFR )
            trainable_pipe = Hyperopt(estimator=planned_pipe, cv=3, max_evals=100, scoring='neg_mean_squared_error')
            trained = trainable_pipe.fit(X_train, y_train)
            print(trained.summary().loss.sort_values().head())
            best_found = trained.get_pipeline()
            pred = trained.predict(X_test)
            print(f'The RMSE of trained pipeline on train set is: {np.sqrt(mse(trained.predict(X_train), y_train))}')
            print(f'The RMSE of trained pipeline on test set is: {np.sqrt(mse(pred, y_test))}')
            ###
            # 1) Save test and predict data to file
            # 2) Save Mae stats to single file
            df_result = pd.DataFrame({"year" : year_test_,
                                      "month": month_test_,
                                      "day": X_test['day'],
                                      "hour": X_test['hour'],
                                      "output_res": y_test,
                                      "output": test_data_raw_,
                                      "prediction": pred})
            fname_split = f.split('/')[-1].split('.')[0] # drop the .csv.autoai
            fname_out = fname_split + '_pred.csv'
            fname_out = path_to_results / subd / fname_out
            df_result.to_csv(fname_out, index=False)
            f_results.write("{},{},{},{}\n".format(fname_split,
                                                mse(pred, y_test),
                                                mae(pred, y_test),
                                                np.sqrt(mse(pred, y_test))))
            print(trained.get_pipeline().pretty_print(show_imports=True, ipython_display=False))
            trained.get_pipeline().pretty_print(ipython_display=True, show_imports=True)

    f_results.close()
