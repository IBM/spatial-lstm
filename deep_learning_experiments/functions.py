import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPooling2D, Masking, AveragePooling2D
import random
from scipy.interpolate import UnivariateSpline,CubicSpline
output = './figs/'
#---------------------------data analysis-------------------------------------
#convert minute data into hourly/daily data by mean
def convert_time(s, period):
    p = 0
    if period == 'hour': p = 60
    elif period == 'day': p = 60*24
    impute_data = []
    i = 0
    while i <len(s):
        impute_data.append(np.mean(s[i:i+p]))
        i += p
    return impute_data

# flatten the batch time series data into a m by n matrix
# m= p.shape[1]: number of sensors, n=p.shape[0]*p.shape[2]: time steps
def flatten(p):
    pred_y_matrix = [[]for _ in range(len(p[0])) ]
    for pp in p:
        a = pp.tolist()
        for m in range(len(a)):
            pred_y_matrix[m] += a[m]
    pred_y_matrix = np.array(pred_y_matrix)
    return np.array(pred_y_matrix).reshape((pred_y_matrix.shape[0], pred_y_matrix.shape[1]))

# split data for univariate time series (one sensor)
# the data is feed to vanilla LSTM model
def single_sensor_split(sensor,t1,t2,period,stride):
    x, y = [], []
    i = 0
    while i+t1+t2+period<=len(sensor):
        x.append(sensor[i:i+t1])
        y.append(sensor[i+t1+period:i+t1+period+t2])
        i += stride
    X = np.array(x).reshape(len(x),len(x[0]),1)
    Y = np.array(y).reshape(len(y),1,len(y[0]))
    return X, Y
# split data for Multivairate time series (matrix of sensors)
# the data is feed to Bidirectional LSTM model

def data_split(dat: list, train_hour: int, test_hour: int, test_period: int,  stride: int):
    '''
    Takes as input an array of list representing a time series (array of vectors). 
    dat: array of list representing time series 
    train_hour: number of lags used for training LSTM (default = 6)
    test_hour: how many predictions we want to make simultaneously (generally == 1)
    test_period: How many timesteps ahead to make prediction (default 47 or 24 hours)
    Stride: How much time to stride ahead by
    
    returns numpy array x,y consisting of features/labels. Three dimensional array
    consisting of [number_of_timesteps, number of sensors, number_of_lags]
    '''
    print('dat shape = ', len(dat), 'and ', len(dat[0]) )
    x, y = [], []
    period = train_hour + test_period + test_hour # This is 6+1+47
    print(f'train_hour = {train_hour}, test_hour = {test_hour}, and test_period = {test_period}' )
    assert(stride != 0) # otherwise get stuck in infinite loop 
    i = 0
    while i + period <= len(dat[0]):
        x_temp = []
        y_temp = []
        for j in range(len(dat)): # loop through each sensor
            x_temp.append(dat[j][i:i + train_hour]) # each line is from i : i+6
            y_temp.append(dat[j][i+ train_hour+ test_period:i+ train_hour+ test_period +test_hour])
        x.append(x_temp)
        y.append(y_temp)
        i += stride
    print(np.array(x).shape, np.array(y).shape)
    return np.array(x), np.array(y)
# intgerpolate the missing values with mask value
def interpolate(data, mask):
    temp = [list(dd) for dd in data]
    d = []
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if temp[i][j] == mask: temp[i][j]= float("NaN")
        df = pd.Series(temp[i]).interpolate(method='linear')
        d.append(df.tolist())
    return d

#Continous data split for masked and unmasked data:
def split_train(Int_dat, Norm_dat,train_time, predict_time,
                                                    predict_position,Stride, start, end, data_name):
    length  = len(Int_dat[0]) # each data frame of same length
    s = int(length*start)
    e = int(length*end)
    #Train =
    Train = [ N[:s]  for N in Norm_dat ]
    Test  = [ M[s:e] for M in Int_dat ]
    print('Training Data Length: ', len(Train),len(Train[0]))
    print('Test Data Length: ', len(Test), len(Test[0]))
    print('Training percentage: ',len(Test[0])/(len(Train[0]))*100,'%' )
    print('Total data size: ', len(Int_dat), len(Int_dat[0]))
    np.savetxt('running_data/' + data_name + '_train.txt', np.exp(np.array(Train)))
    np.savetxt('running_data/' + data_name + '_test.txt', np.exp(np.array(Test)))
    train_x, train_y = data_split(Train, train_time, predict_time,
                                                    predict_position,Stride)
    test_x, test_y = data_split(Test, train_time, predict_time,
                                                    predict_position,Stride)
    return train_x, train_y, test_x, test_y
# Data normalization:
# using log function and skip the masked data with -1
# 0 values are replaced by 1e-5 in order to avoid nan value in log
def data_normalize(Dat):
    new_dat  = []
    for d in Dat:
        min, max = np.min(sorted(list(set(d)))[1]), np.max(d)
        a = max -min
        temp = []
        for val in d:
            if val == -1: temp.append(val)
            else:
                norm =np.log(val +1e-3 )
                temp.append(norm ) #temp.append((val - min)/a )
        new_dat.append(temp)
    return new_dat

#---------------------------MAE-------------------------------------
def log_mae(py,ty, t,plot_name):
    print('predict data size: ',len(py),len(py[0]))
    print('exact data size: ', len(ty), len(ty[0]))
    mae_lis = []
    std = []
    for i in range(len(py)):
        mae = np.exp(np.array(py[i])) - np.exp(np.array(ty[i]))
        print('Predicted Data Point size for sensor '+str(i+1)+' ', len(mae))
        mae = np.abs(mae)
        mae_lis.append(np.mean(np.abs(mae)))
        std.append(np.std(mae))
    return mae_lis, std
#---------------------------Plot-------------------------------------
def py_ty_plot(data_py, data_ty, s, p,data_name, idx):
    l = len(data_py[0])
    x = range(len(data_py[0]))
    y = data_py[idx]
    xs = np.linspace(0, l, p)
    ss = UnivariateSpline(x, y)
    ys = ss(xs)
    plt.figure()
    plt.scatter(range(l),y ,s = s)
    plt.plot(xs, ys,label = 'Sp')
    y = data_ty[idx]
    ss = UnivariateSpline(x, y)
    ys = ss(xs)
    plt.plot(xs, ys,label = 'Exact')
    plt.scatter(range(l),data_ty[idx],s = s)
    #plt.title('Oxygen Data at sensor No.: '+ str(idx))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
              ncol=3, fancybox=True, shadow=True)
    plt.title('Sensor No: '+str(idx+1))
    plt.savefig(output + data_name+'_Time_series_sensor_No'+str(idx)+'.pdf',bbox_inches='tight')
    plt.show()

# Heatmap of two matrix
def multi_heatmap(Test_y, Pred_y, t,plot_name, plot):
    Test_y = Test_y.reshape(Test_y.shape[0],Test_y.shape[1],Test_y.shape[2])
    Pred_y = Pred_y.reshape(Pred_y.shape[0],Pred_y.shape[1],Pred_y.shape[2])
    py = flatten(Pred_y)
    ty = flatten(Test_y)
    min, max = np.min(ty), np.max(ty)
    #-------------------------------Plot the new heatmap of predict data vs test data
    if plot:
        plt.figure()
        print(len(py),len(ty))
        ax1 = sns.heatmap(np.array(ty).T,vmin = min, vmax = max)
        ax1.set_title('Exact Data')
        ax1.set(xlabel='Sensors', ylabel='Time Step')
        f1 = ax1.get_figure()
        f1.savefig(output + str(plot_name) + '_Exact_heatmap.pdf',bbox_inches='tight')
        plt.show()
        plt.figure()
        ax2 = sns.heatmap(np.array(py).T,vmin = min, vmax = max)
        ax2.set_title('Predicted Data')
        ax2.set(xlabel='Sensors', ylabel='Time Step')
        f2 = ax2.get_figure()
        f2.savefig(output + str(plot_name) + '_Predicted_heatmap.pdf',bbox_inches='tight')
        plt.show()
        plt.figure()
        error = np.array(py).T - np.array(ty).T
        print('error shape: ', error.shape)
        ax3 = sns.heatmap(error,vmin = min, vmax = max)
        ax3.set_title('Error Map')
        ax3.set(xlabel='Sensors', ylabel='Time Step')
        f3 = ax3.get_figure()
        f3.savefig(output + str(plot_name) + '_Error_heatmap.pdf',bbox_inches='tight')
        plt.show()
    MAE_lis, STD_lis = log_mae(py,ty, t,plot_name)
    return MAE_lis, STD_lis
#---------------------------Model-------------------------------------

# Bidirectional LSTM model
def stacked_LSTM(X, Y, name, epochs=50, batch_size=256):
    time_step = X.shape[1]
    input_dim =  X.shape[2]
    out =  Y.shape[2]
    #Bidirectional LSTM
    start = time.time()
    model = Sequential()
    #model.add(Masking(mask_value=-1.,input_shape=(time_step, input_dim)))
    model.add(Bidirectional(LSTM(64,activation='relu', input_shape=(time_step, input_dim),return_sequences=True)))
    model.add(Bidirectional(LSTM(32, activation='relu', input_shape=(time_step, input_dim), return_sequences=True)))
    #model.add(Masking(mask_value=-1.,input_shape=(time_step, input_dim)))
    model.add(Dense(out))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    print(f'Data size for features {X.shape} and labels {Y.shape}')
    print(f'input shape = {time_step, input_dim}')
    callback = callbacks.EarlyStopping(monitor='loss', patience=3)
    hist = model.fit(X, Y,epochs=epochs, validation_split=.2,
              verbose=1, batch_size=batch_size, callbacks=[callback])
    model.summary()
    with open('./model_summary/' + name + '_SPATIALmodelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    end = time.time()
    print("Total compile time: --------", end - start, 's')
    return model, hist

def baseline_CNN(X,Y, name, epochs=50, batch_size=256):
    time_step = X.shape[1] #X.shape[1]
    input_dim = X.shape[2] #X.shape[2]
    out = Y.shape[2] # Y.shape[2]
    #CNN
    start = time.time()
    model = Sequential()
    #model.add(Masking(mask_value=-1.,input_shape=(time_step, input_dim)))
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu', input_shape=(time_step, input_dim, 1)))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    model.add(AveragePooling2D((1, input_dim - out + 1), padding="valid"))
    #model.add(Flatten())
    model.add(Dense(out))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    callback = callbacks.EarlyStopping(monitor='loss', patience=3)
    hist = model.fit(X, Y, epochs=epochs, validation_split=.2,
              verbose=1, batch_size=batch_size, callbacks=[callback])
    model.summary()
    with open('./model_summary/' + name + '_CNNmodelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    end = time.time()
    print("Total compile time: --------", end - start, 's')
    return model, hist

def univariate_CNN(X,Y, name, epochs=50, batch_size=256):
    time_step = X.shape[0]
    input_dim = X.shape[1]
    out = 1
    #CNN
    print(f'Time step = {time_step} and num_features = {input_dim}')
    print(X.shape, Y.shape)
    start = time.time()
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(input_dim, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='relu')) # we make one shot prediction
    model.compile(loss='mean_absolute_error', optimizer='adam')
    callback = callbacks.EarlyStopping(monitor='loss', patience=3)
    hist = model.fit(X, Y, epochs=epochs, validation_split=.2,
              verbose=1, batch_size=batch_size, callbacks=[callback])
    model.summary()
    with open('./model_summary/' + name + '_CNNmodelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    end = time.time()
    print("Total compile time: --------", end - start, 's')
    return model, hist

def baseline_LSTM(X,Y, name, epochs=50, batch_size=256):
    print(f'At baseline LSTM shape X ={X.shape} and Y = {Y.shape}')
    time_step = X.shape[1]
    input_dim = 1
    out = Y.shape[1]
    print(X.shape, Y.shape)
    start = time.time()
    model = Sequential()
    #model.add(Masking(mask_value=-1.,input_shape=(time_step, input_dim)))
    model.add(LSTM(16, activation='relu', input_shape=(time_step,1), return_sequences=True))
    model.add(LSTM(16, activation='relu'))
    #model.add(Flatten())
    model.add(Dense(out))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    callback = callbacks.EarlyStopping(monitor='loss', patience=3)
    hist = model.fit(X, Y, epochs=epochs, validation_split=.2,
              verbose=1, batch_size=batch_size, callbacks=[callback])
    model.summary()
    with open('./model_summary/' + name + '_LSTM_modelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    end = time.time()
    print("Total compile time: --------", end - start, 's')
    return model, hist

# Spatial Bidiretional Model
def SP_Learner(data, data_name,  train_split=0.8, test_split=1, train_time=6, 
                predict_time=1, predict_position=47, Stride=1,
                Norm=0, epoches=50, run_model="LSTM",batch_size=256,  plot=False):
    '''
    data: list(np.array()) a list of dataframes containing a numpy array of sensor data
    train_time: number of lags used for training LSTM
    predict_time: plotting prediction values
    predict_position: How many timesteps ahead to make prediction
    Stride: How much time to stride ahead by
    train_split: Train split (default = 0.8)
    test_split: Test split (default = 1)
    data_name: file name to save data
    Norm: whether data is differenced or not
    epoches: training epochs
    run_model: == 'SPATIAL', 'CNN', or 'LSTM'
    '''
    print('########################Start##################################')
    norm_dat = data
    norm_int_dat = data
    if Norm == 1: # predicting difference
        print('we act on differences')
        norm_dat = data_normalize(data)
        norm_int_dat = interpolate(norm_dat, -1)
        print(len(norm_dat), len(norm_int_dat))
    else:
        print('we are acting on raw data', Norm)
        print(len(norm_dat), len(norm_int_dat))

    #-----------------------------------plot--------------------------
    if plot:
        f1 = sns.heatmap(norm_dat)
        f1.set_title(data_name + ' Masked Data')
        plt.figure()
        plt.show()
        f2 = sns.heatmap(norm_int_dat)
        f2.set_title(data_name + ' Interpolated Data')
        plt.figure()
        plt.show()
    train_x, train_y, test_x, test_y = split_train(norm_int_dat, norm_dat, train_time, predict_time,
                                                    predict_position, Stride,train_split, test_split, data_name)
    print('Train data size(batch, row, column)',train_x.shape, train_y.shape)
    print('test data size(batch, row, column)',test_x.shape, test_y.shape)
    if run_model == 'CNN':
        # select whether to run bidirectional LSTM
        # or baseline CNN model
        print('begin baseline CNN')
        pred_y = []
        for i in range(0, train_x.shape[1]):  # loop over all sensors
            print(f'shape of train_x is {train_x.shape} and train_y = {train_y.shape} and {len(train_y)}')
            model, hist = univariate_CNN(train_x[:,i,:],train_y[:,i,:], data_name, epoches, batch_size) 

            pred_y.append(model.predict(test_x[:,i,:], verbose=1))
        pred_y = np.array(pred_y)
        pred_y = pred_y.reshape((pred_y.shape[1], pred_y.shape[0], pred_y.shape[2]))
        # model, hist = baseline_CNN(train_x,train_y, data_name, epoches)
        # pred_y = model.predict(test_x, verbose = 1)
    elif run_model == 'SPATIAL': 
        print('Begin Stacked LSTM training')
        # We use the same process to run whether it 
        model, hist = stacked_LSTM(train_x,train_y, data_name, epoches, batch_size) 
        pred_y = model.predict(test_x, verbose = 1)
    else: # run lstm model
        pred_y = []
        for i in range(0, train_x.shape[1]):  
            print(f'shape of train_x is {train_x.shape} and train_y = {train_y.shape} and {len(train_y)}')
            model, hist = baseline_LSTM(train_x[:,i,:],train_y[:,i,:], data_name, epoches, batch_size) 

            pred_y.append(model.predict(test_x[:,i,:], verbose=1))
        pred_y = np.array(pred_y)
        pred_y = pred_y.reshape((pred_y.shape[1], pred_y.shape[0], pred_y.shape[2]))

    print('Dimensions of the output data =' , pred_y.shape, test_y.shape)
    error, std = multi_heatmap(test_y, pred_y, predict_time,data_name, plot)
    py = flatten(pred_y)
    ty = flatten(test_y)
    plt.figure()
    print('shapes', np.array(ty).shape, np.array(py).shape)
    if plot:
        for i in range(len(ty)):
            plt.scatter(range(len(ty[i])),ty[i]-py[i])
        plt.title(data_name + ' Test Errors')
    print('MAE: ', np.mean(error), 'STD: ',np.mean(std))
    print('########################End##################################')
    return np.array(py), np.array(ty), error, std, model

