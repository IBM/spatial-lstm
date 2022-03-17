## This script contains utility functions used for the SPATIAL,
## Lale, and AMLP model results. 

combine_spatial_and_observation <- function(fname_raw, fname_spatial, ADCP){
  ## Function to read the raw and Spatial results and join to dataframe
  df_raw = read.table(fname_raw, sep=',', header = TRUE)
  if (ADCP){
    print('we;re doing ADCP')
    df_raw$observed_timestamp <- as.POSIXct(strptime(df_raw$observed_timestamp, "%Y-%m-%d %H:%M:%S", tz="UTC"))
  } else{
    df_raw$observed_timestamp <- as.POSIXct(strptime(df_raw$observed_timestamp, "%Y-%m-%d %H:%M:%S+00:00", tz="UTC"))
  }
  print(head(df_raw,3))
  df_spat = read.table(fname_spatial, sep=',', header = TRUE)
  if (ADCP){
    print('were doing ADCP again')
    df_spat$date <- as.POSIXct(strptime(df_spat$date, "%Y-%m-%d %H:%M:%S", tz="UTC"))
  } else {
    df_spat$date <- as.POSIXct(strptime(df_spat$date, "%Y-%m-%d %H:%M:%S+00:00", tz="UTC"))
  }
  print(head(df_spat,3))
  df_jn <- merge(df_spat,df_raw, by.x='date', by.y = 'observed_timestamp')
  names(df_jn)[names(df_jn) == "value.x"] <- "SPATIAL"
  names(df_jn)[names(df_jn) == "value.y"] <- "observation"
  #  df_jn$diff_spatial = df_jn$observation - df_jn$SPATIAL
  df_jn
}

add_lale_and_amlp_to_df <- function(df_, df_lale, df_amlp){
  df_lale <- add_minute(df_lale)
  df_lale$date <- ISOdate(df_lale$year, df_lale$month, df_lale$day, 
                          df_lale$hour, df_lale$minute, tz = 'UTC')
  df_lale$date <- df_lale$date + 86400 
  df_lale$amlp <- df_amlp$pred
  df_jn <- merge(df_,df_lale, by.x='date', by.y = 'date')
  names(df_jn)[names(df_jn) == "prediction"] <- "Lale"
  #  df_jn$diff_lale = df_jn$observation - df_jn$SPATIAL
  df_jn
}

add_minute <- function(df){
  # The Lale and AMLP results don't include minutes so add manually
  df$minute <- 0
  for (i in seq(1, dim(df)[1])){
    if (i %% 2 == 0){
      df$minute[i] <- 30 
    }
  }
  df
}

add_observed_to_residual <- function(df_sensor, df_res){
  ### The residual is created by taking the difference, 
  ### lagged by the forecast horizon. 
  ### ie y_res = y_t - y_t-48
  ### reconstruction is the reverse
  
  # first find date of first index by matching timestamps
  ind_beg <- match(df_res$date[1], df_sensor$observed_timestamp)
  df_res$reconstruct_signal <- NULL
  df_res$test_reconstruct <- NULL
  for (i in seq(1, dim(df_res)[1])){
    df_res$reconstruct_signal[i] <- df_res$prediction[i] + df_sensor$value[ind_beg + i -48]
    df_res$test_reconstruct[i] <- df_res$output_res[i] + df_sensor$value[ind_beg + i -48]
  }
  return(df_res)
}

add_observed_to_residual_spat <- function(df_sensor, df_res){
  ### The residual is created by taking the difference, 
  ### lagged by the forecast horizon. 
  ### ie y_res = y_t - y_t-48
  ### reconstruction is the reverse
  
  # first find date of first index by matching timestamps
  ind_beg <- match(df_res$date[1], df_sensor$observed_timestamp)
  df_res$reconstruct_signal <- NULL
  df_res$test_reconstruct <- NULL
  df_res$output <- NULL
  df_res$output_res <- NULL
  for (i in seq(1, dim(df_res)[1])){
    df_res$reconstruct_signal[i] <- df_res$value[i] + df_sensor$value[ind_beg + i -48]
    df_res$output[i] <-  df_sensor$value[ind_beg + i]
  }
  return(df_res)
}



add_observed_to_residual_amlp <- function(df_sensor, df_res){
  ### The residual is created by taking the difference, 
  ### lagged by the forecast horizon. 
  ### ie y_res = y_t - y_t-48
  ### y_t = y_res + y_t-48
  ### reconstruction is the reverse
  
  # first find date of first index by matching timestamps
  ind_beg <- match(df_res$date[1], df_sensor$observed_timestamp)
  diff_sens <- diff(df_sensor$value, lag=48)
  df_res$reconstruct_signal <- NULL
  df_res$output <- NULL
  df_res$output_res <- NULL
  
  for (i in seq(1, dim(df_res)[1])){
    df_res$reconstruct_signal[i] <- df_res$pred[i] + df_sensor$value[ind_beg + i -48]
    df_res$output[i] <-  df_sensor$value[ind_beg + i]
    df_res$output_res[i] <- diff_sens[ind_beg  + i]
  }
  return(df_res)
}



reconstruct_residual <- function(sensor_fname, residual_fname, ADCP){
  ## Function to read the raw and Spatial results and join to dataframe
  df_sensor = read.table(sensor_fname, sep=',', header = TRUE)
  if (ADCP){
    print('we;re doing ADCP')
    df_sensor$observed_timestamp <- as.POSIXct(strptime(df_sensor$observed_timestamp, "%Y-%m-%d %H:%M:%S", tz="UTC"))
  } else{
    df_sensor$observed_timestamp <- as.POSIXct(strptime(df_sensor$observed_timestamp, "%Y-%m-%d %H:%M:%S+00:00", tz="UTC"))
  }
  df_res = read.table(residual_fname, sep=',', header = TRUE)
  ## If Lale or AMLP, date is of form, year, month, day, etc.
  ## Also, minute is not included so need to add manually
  df_res <- add_minute(df_res)
  df_res$date <- ISOdate(df_res$year, df_res$month, df_res$day, 
                         df_res$hour, df_res$minute, tz = 'UTC')
  ## For Lale or AMLP, date refers to the forecast date while senor refers to 
  ## date t_now, so need to step forward 24 hours
  df_res$date <- df_res$date + 86400 
  ### 
  ###
  df_res = add_observed_to_residual(df_sensor, df_res)
  
  return(list("df_sensor" = df_sensor, "df_model" =  df_res))
}


reconstruct_residual_spat <- function(sensor_fname, residual_fname, ADCP){
  ## Function to read the raw and Spatial results and join to dataframe
  df_sensor = read.table(sensor_fname, sep=',', header = TRUE)
  if (ADCP){
    print('we;re doing ADCP')
    df_sensor$observed_timestamp <- as.POSIXct(strptime(df_sensor$observed_timestamp, "%Y-%m-%d %H:%M:%S", tz="UTC"))
  } else{
    df_sensor$observed_timestamp <- as.POSIXct(strptime(df_sensor$observed_timestamp, "%Y-%m-%d %H:%M:%S+00:00", tz="UTC"))
  }
  df_res = read.table(residual_fname, sep=',', header = TRUE)
  ## If Lale or AMLP, date is of form, year, month, day, etc.
  ## Also, minute is not included so need to add manually
  df_res$date <- as.POSIXct(strptime(df_res$date, "%Y-%m-%d %H:%M:%S", tz="UTC")) 
  ## For Lale or AMLP, date refers to the forecast date while senor refers to 
  ## date t_now, so need to step forward 24 hours
  ### 
  ###
 # df_res$date <- df_res$date - 86400
  df_res = add_observed_to_residual_spat(df_sensor, df_res)
  
  return(list("df_sensor" = df_sensor, "df_model" =  df_res))
}





reconstruct_residual_amlp <- function(sensor_fname, residual_fname, ADCP){
  ## Function to read the raw and AMLP results and join to dataframe
  df_sensor = read.table(sensor_fname, sep=',', header = TRUE)
  if (ADCP){
    print('we;re doing ADCP')
    df_sensor$observed_timestamp <- as.POSIXct(strptime(df_sensor$observed_timestamp, "%Y-%m-%d %H:%M:%S", tz="UTC"))
  } else{
    df_sensor$observed_timestamp <- as.POSIXct(strptime(df_sensor$observed_timestamp, "%Y-%m-%d %H:%M:%S+00:00", tz="UTC"))
  }
  df_res = read.table(residual_fname, sep=',', header = TRUE)
  ## AMLP has no date signal so need to reconstruct from test file
  df_res <- add_minute(df_res)
  df_res$date <- ISOdate(df_res$year, df_res$month, df_res$day, 
                         df_res$hour, df_res$minute, tz = 'UTC')
  ## For Lale or AMLP, date refers to the forecast date while senor refers to 
  ## date t_now, so need to step forward 24 hours
  ## For AMLP it's a little complicated.
  ## We start with the the data processed by Paulito AutoML 
  ## which time aligns data along
  ## timestamp_t, output_t+48
  ## Then we compute diff based on diff(data, nlag = 48)
  ## which computes of form
  ## ==>  x[(1+lag):n] - x[1:(n-lag)]
  ## This steps data forward again so is of form
  ## timestamp_t, output_t+96. Hence, need to update this
 ##  df_res$date <- df_res$date + (86400 * 2) 
  ### 
  ###
  df_res = add_observed_to_residual_amlp(df_sensor, df_res)

  return(list("df_sensor" = df_sensor, "df_model" =  df_res))
}

combine_all_forecasts <- function(df_lale, df_amlp, df_spat){
  df_lale_ <- df_lale[c('date', 'reconstruct_signal', 'output')]
  df_amlp_ <- df_amlp[c('date', 'reconstruct_signal', 'output')]
  df_spat_ <- df_spat[c('date', 'reconstruct_signal', 'output')]
  df_lale_$date <- df_lale_$date - 1800
  names(df_lale_)[names(df_lale_) == "reconstruct_signal"] <- "AutoAI"
  names(df_amlp_)[names(df_amlp_) == "reconstruct_signal"] <- "AMLP"
  names(df_spat_)[names(df_spat_) == "reconstruct_signal"] <- "SPATIAL"
  
  df  <- merge(df_lale_,df_amlp_, by.x='date', by.y = 'date')
  df  <- merge(df,df_spat_, by.x='date', by.y = 'date')
  names(df)[names(df) == "output"] <- "observation"
  df
}
