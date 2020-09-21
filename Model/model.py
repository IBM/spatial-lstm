import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.layers import  Masking
from scipy.interpolate import UnivariateSpline,CubicSpline
output = 'C:/Users/yihao/Dropbox/Research/ADCP/figs/'
#---------------------------data analysis-------------------------------------
# flattern the batch time series data into a m by n matrix
# m= p.shape[1]: number of sensors, n=p.shape[0]*p.shape[2]: time steps
def flattern(p):
    pred_y_matrix = [[]for _ in range(len(p[0])) ]
    for pp in p:
        a = pp.tolist()
        for m in range(len(a)):
            pred_y_matrix[m] += a[m]
    return pred_y_matrix


# interpolate the missing values with mask value
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
def split_train(Int_dat, Norm_dat,T1,T2,T3,Stride, start, end,data_name):
    length  = len(Int_dat[0])
    s = int(length*start)
    e = int(length*end)
    Train = [ N[:s] + N[e:] for N in Norm_dat ]
    Test  = [ M[s:e] for M in Int_dat ]
    print('Training Data Length: ', len(Train),'X',len(Train[0]))
    print('Test Data Length: ', len(Test), 'X',len(Test[0]))
    print('Testing percentage: ',len(Test[0])/(len(Test[0])+len(Train[0]))*100,'%' )
    print('Total data size: ', len(Int_dat), 'X', len(Int_dat[0]))
    np.savetxt(data_name + '_train.txt', np.exp(np.array(Train)))
    np.savetxt(data_name + '_test.txt', np.exp(np.array(Test)))
    train_x, train_y = data_split(Train, T1, T2, T3,Stride)
    test_x, test_y = data_split(Test, T1, T2, T3,Stride)
    return train_x, train_y, test_x, test_y

# split data for Multivairate time series (matrix of sensors)
# the data is feed to Bidirectional LSTM model
def data_split(dat, train_hour, test_hour, predict_position,  stride):
    #train_hour: training data length
    #test_hour: testing data length
    #predict_position: gap between train_hour and test_hour
    x, y = [], []
    period = train_hour + predict_position + test_hour
    i = 0
    while i + period <= len(dat[0]):
        x_temp = []
        y_temp = []
        for j in range(len(dat)):
            x_temp.append(dat[j][i:i + train_hour])
            y_temp.append(dat[j][i+ train_hour+ predict_position:i+ train_hour+ predict_position +test_hour])
        x.append(x_temp)
        y.append(y_temp)
        i += stride
    return np.array(x), np.array(y)

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
                norm =np.log(val) if val!=0 else 1e-5
                temp.append(norm ) #temp.append((val - min)/a )
        new_dat.append(temp)
    return new_dat

#---------------------------MAE STD-------------------------------------
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
    plt.title('Sensor No: ' + str(idx + 1))
    plt.scatter(range(l),y ,s = s)
    plt.plot(xs, ys,label = 'Sp')
    y = data_ty[idx]
    ss = UnivariateSpline(x, y)
    ys = ss(xs)
    plt.plot(xs, ys,label = 'Exact')
    plt.scatter(range(l),data_ty[idx],s = s)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
              ncol=3, fancybox=True, shadow=True)
    plt.savefig(output + data_name+'_Time_series_sensor_No'+str(idx)+'.pdf',bbox_inches='tight')
    plt.show()
    return

# Heatmap of two matrix
def multi_heatmap(Test_y, Pred_y, t,plot_name):
    Test_y = Test_y.reshape(Test_y.shape[0],Test_y.shape[1],Test_y.shape[2])
    Pred_y = Pred_y.reshape(Pred_y.shape[0],Pred_y.shape[1],Pred_y.shape[2])
    py = flattern(Pred_y)
    ty = flattern(Test_y)
    min, max = np.min(ty), np.max(ty)
    #Plot the new heatmap of predict data vs test data
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
    #return the mae and std after plot
    MAE_lis, STD_lis = log_mae(py,ty, t,plot_name)
    return MAE_lis, STD_lis

#---------------------------Model-------------------------------------
# Bidirectional LSTM model
def stacked_LSTM(X, Y):
    time_step = X.shape[1]
    input_dim = X.shape[2]
    out = Y.shape[2]
    #Bidirectional LSTM
    start = time.time()
    model = Sequential()
    model.add(Masking(mask_value=-1.,input_shape=(time_step, input_dim)))
    model.add(Bidirectional(LSTM(32,activation='elu', input_shape=(time_step, input_dim),return_sequences=True)))
    #model.add(Bidirectional(LSTM(64, activation='elu', input_shape=(time_step, input_dim), return_sequences=True)))
    #model.add(Masking(mask_value=-1.,input_shape=(time_step, input_dim)))
    model.add(Dense(out))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    hist = model.fit(X, Y,epochs=100, validation_split=.2,
              verbose=0, batch_size=10)
    model.summary()
    end = time.time()
    print("Total compile time: --------", end - start, 's')
    return model, hist

def SP_Learner(data, train_time, predict_time, predict_position,Stride, start, end, data_name):
    print('########################Start##################################')
    norm_dat = data_normalize(data) #normalized data (used for training)
    norm_int_dat = interpolate(norm_dat,-1) #normalized interpolated data (used for prediction)
    #-----------------------------------plot-------------------------------------------
    f1 = sns.heatmap(norm_dat)
    f1.set_title(data_name + ' Masked Data')
    plt.figure()
    plt.show()
    f2 = sns.heatmap(norm_int_dat)
    f2.set_title(data_name + ' Interpolated Data')
    plt.figure()
    plt.show()
    # -----------------------------------end plot-------------------------------------------
    # split the data set
    train_x, train_y, test_x, test_y = split_train(norm_int_dat, norm_dat, train_time, predict_time,predict_position, Stride,start, end, data_name)
    print('Train data size(batch, row, column)','Train X: ', train_x.shape,' ,Train Y: ',train_y.shape)
    print('test data size(batch, row, column)','Test X: ',test_x.shape,' ,Test Y: ',test_y.shape)
    # model training
    model, hist = stacked_LSTM(train_x,train_y)
    pred_y = model.predict(test_x, verbose=1)
    error, std = multi_heatmap(test_y, pred_y,predict_time,data_name)
    py = flattern(pred_y)
    ty = flattern(test_y)
    plt.figure()
    for j in range(len(ty)):
        plt.scatter(range(len(ty[j])),[ty[j][i]-py[j][i] for i in range(len(ty[j]))])
    plt.title(data_name + ' Test Errors')
    print('MAE: ', np.mean(error), 'STD: ',np.mean(std))
    print('########################End##################################')
    return py, ty, error, std, model
