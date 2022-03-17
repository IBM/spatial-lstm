### Process the SPATIAL experiments into same format as Lale/AMLP
library(here)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(dplyr)
library(gridExtra)
library(scales)
this.dir <- dirname(parent.frame(2)$ofile) # frame(3) also works.

setwd(this.dir)
print(getwd())
source('./util_fncs.R')
mae_all = c()

create_grid_plot = FALSE

results_dir = './figures/'
fout = paste(results_dir, 'MAE_all_models_normalised.csv', sep = "")
fout_mae = paste(results_dir, 'MAE_each_station.csv', sep = "")
fout_mape = paste(results_dir, 'MAPE_each_station.csv', sep = "")

## List the pertinent directories that store data, for SPATIAL
## results, amlp results, and lale results...
# input data (observation data)
raw_data_dir = '../sensor_data/interpolated_data/'
## Spatial data dir
dl_dir = '../deep_learning_experiments/prediction/'
# Lale data dire
lale_dir = '../machine_learning_experiments//autoAI_experiments//results/'
# AMLP dir
amlp_dir = '../machine_learning_experiments/amlp/results/'

## Load the raw and SPATIAL results.
### Files we use for visualisation (for temperature and oxygen 1--9, for ADCP 1-4) 
sensors_to_use <- list(c(1,8,9), c(1,8,9), c(2,4)) # for temperature, oxy, and adcp respectively
sensors_to_use <- list(c(9), c(9), c(4)) # for temperature, oxy, and adcp respectively

write_stats = TRUE
if (length(unlist(sensors_to_use[1])) == 1){
  write_stats = FALSE # We use this just to make plots
}
# data file stem names for the 3
sensor_stem_list <- c('Temp_Sensor_', 'Oxygen_Sensor_', 'ADCP_Sensor_')


# Different subdirectory details
lale_sensor_dir = c('temperature/', 'oxygen/', 'adcp/')
amlp_sensor_dir = c('temp/', 'oxygen/', 'adcp/')

# Some simple annotation details for plotting
yax_labs <- c(expression('Temperature ('*~degree*C*')'), 'DO (mg/L)', 'Speed (cm/s)')
annotate_xlox <- c('2018-12-15', '2018-12-15', '2019-11-01')
annotate_yloc <- list(c(7.75, 7.25, 6.75), c(7.75, 7.25, 6.75), c(45, 40, 35))

lims_arr <- list(c("2018-11-27","2018-12-11"),
             c("2018-11-27","2018-12-11"),
              c("2019-10-22","2019-11-06"))

lims_arr <- list(c("2018-11-27","2018-12-11"),
                 c("2018-11-27","2018-12-11"),
                 c("2019-12-10","2019-12-30"))

## We want to read both the Lale and raw data and recreate the true signal 
## from residual forecast (Lale)
## Basic function is of form
## y_reconstruct = y_sens(t-48) + y_res(t)
#1) Read the raw and Lale data

for (sens_type in seq(3,3)){ #3) ){ # itereate across temperature, oxygen, and ADCP
  mae_lale = c();   mape_lale = c()
  mae_amlp = c();   mape_amlp = c()
  mae_spatial = c(); mape_spatial =c()
  mae_cnn = c(); mape_cnn =c()
  mae_lstm = c(); mape_lstm =c()
  
  id_loop <- 1
  sens_extract = unlist(sensors_to_use[sens_type]) # IDs of which sensors to use for each of temp, oxy, adcp
  sensor_stem = sensor_stem_list[sens_type]
  raw_data_fname <-  paste(paste(sensor_stem_list[sens_type], 
                                 sens_extract, sep=''), '.csv', sep = '')
  # 2) spatial data files
  spatial_fname <-  paste(paste(sensor_stem_list[sens_type], 
                                sens_extract, sep=''), '_SPATIAL_prediction.csv', sep = '')
  # 3) Lale data files
  lale_fname <- paste(paste(sensor_stem_list[sens_type], 
                            sens_extract, sep = ''), '_pred.csv', sep='')
  # 4) AMLP data files
  amlp_fname <- paste(paste(sensor_stem_list[sens_type], 
                            sens_extract, sep = ''), '.prediction', sep='')
  amlp_fname_test <- paste(paste(sensor_stem_list[sens_type], 
                            sens_extract, sep = ''), '.test', sep='')
  
  # 5) CNN data files
  cnn_fname <-  paste(paste(sensor_stem_list[sens_type], 
                                sens_extract, sep=''), '_CNN_prediction.csv', sep = '')
  
  # 6) LSTM data files
  lstm_fname <-  paste(paste(sensor_stem_list[sens_type], 
                                sens_extract, sep=''), '_LSTM_prediction.csv', sep = '')
  
  
  ADCP_FLAG = FALSE # We need to treat ADCP slightly differently
  for (id_loop in seq(1, length(sens_extract))) # we generally select across 3 sensors for each dataset.
    {                                           # The exact ID is defined above
    sensor_fname = here(getwd(), raw_data_dir, raw_data_fname[id_loop])
    residual_fname = here(getwd(),lale_dir, lale_sensor_dir[sens_type], lale_fname[id_loop])
    amlp_fname_ = here(getwd(),amlp_dir, amlp_sensor_dir[sens_type], amlp_fname[id_loop])
    spat_fname_ = here(getwd(),dl_dir, spatial_fname[id_loop])
    cnn_fname_ = here(getwd(),dl_dir, cnn_fname[id_loop])
    lstm_fname_ = here(getwd(),dl_dir, lstm_fname[id_loop])
    print(sensor_fname)  
    print(residual_fname)
    if (sens_type == 3){ADCP_FLAG=TRUE}
    list_df = reconstruct_residual(sensor_fname, residual_fname, ADCP=ADCP_FLAG)
    list_df_amlp = reconstruct_residual_amlp(sensor_fname, amlp_fname_, ADCP=ADCP_FLAG)
    list_df_spat = reconstruct_residual_spat(sensor_fname, spat_fname_, ADCP=ADCP_FLAG)
    list_df_cnn = reconstruct_residual_spat(sensor_fname, cnn_fname_, ADCP=ADCP_FLAG)
    list_df_lstm = reconstruct_residual_spat(sensor_fname, lstm_fname_, ADCP=ADCP_FLAG)
    
    df_sensor = list_df[['df_sensor']]
    df_lale = list_df[['df_model']]
    df_amlp = list_df_amlp[['df_model']]
    df_spat = list_df_spat[['df_model']]
    df_cnn = list_df_cnn[['df_model']]
    df_lstm = list_df_lstm[['df_model']]
    if (sens_type == 1){ ## filter out crazy high value for temp
      df_lale$reconstruct_signal[df_lale$reconstruct_signal>8]  = NA
      df_lale$output[df_lale$output>8]  = NA
      df_amlp$reconstruct_signal[df_amlp$reconstruct_signal>8]  = NA
      df_amlp$output[df_amlp$output>8]  = NA
      df_spat$reconstruct_signal[df_spat$reconstruct_signal>8]  = NA
      df_spat$output[df_spat$output>8]  = NA
      df_cnn$reconstruct_signal[df_cnn$reconstruct_signal>8]  = NA
      df_cnn$output[df_cnn$output>8]  = NA
      df_lstm$reconstruct_signal[df_lstm$reconstruct_signal>8]  = NA
      df_lstm$output[df_lstm$output>8]  = NA
    }
    
    if (sens_type == 1){ ##and crazy low values
      lim_b <- 4.5
      date_beg_lale = as.POSIXct("2018-12-05","%Y-%m-%d", tz = "UTC") 
      df_spat <- df_spat %>% filter((date < date_beg_lale & 
                                       reconstruct_signal>lim_b)
                                    | date > date_beg_lale)
      df_cnn <- df_cnn  %>% filter((date < date_beg_lale & 
                                       reconstruct_signal>lim_b)
                                    | date > date_beg_lale)
      df_lstm <- df_lstm %>% filter((date < date_beg_lale & 
                                       reconstruct_signal>lim_b)
                                    | date > date_beg_lale)
  
    }
    if (sens_type == 3){ ##and crazy low values
      lim_b <-23
      date_beg_lale = as.POSIXct("2019-12-10","%Y-%m-%d", tz = "UTC") 
      df_spat <- df_spat %>% filter((date < date_beg_lale & 
                                       reconstruct_signal>lim_b)
                                    | date > date_beg_lale)
      
    }
    ylab <- expression('Temperature ('*~degree*C*')')
    ylab <- yax_labs[sens_type]
   
    p_ts_total <- ggplot(data=df_lale,aes(x=as.POSIXct(date),y=output, color = 'sensor')) +
      geom_point(size = 0.5)  +
      geom_line(data=df_lale, aes(x = as.POSIXct(date), y = reconstruct_signal ,color = 'AutoAI') ) +
      geom_line(data=df_amlp, aes(x = as.POSIXct(date), y = reconstruct_signal ,color = 'AMLP') ) +
      
      geom_line(data=df_cnn, aes(x = as.POSIXct(date), y = reconstruct_signal ,color = 'CNN') ) +
      geom_line(data=df_lstm, aes(x = as.POSIXct(date), y = reconstruct_signal ,color = 'LSTM') ) +
      geom_line(data=df_spat, aes(x = as.POSIXct(date), y = reconstruct_signal ,color = 'SPATIAL') ) +
      
      labs(x = "", 
           y = ylab, 
           'colour' = "")
    # unify the color legend
    p_ts_total <- p_ts_total + scale_colour_manual(values=c(sensor = "black",
                                                            AutoAI = "#ff6e54",
                                                            AMLP="#ffa600",
                                                            CNN="#444e86",
                                                            LSTM="#955196",
                                                            SPATIAL="#488f31"))
    
    
    lims <- as.POSIXct(strptime(unlist(lims_arr[sens_type]), format = "%Y-%m-%d"))
#    p_ts_total <- p_ts_total +  scale_x_datetime(limits =  lims, labels=date_format("%d-%m-%Y"))
#    p_ts_res <- p_ts_res +  scale_x_datetime(limits =  lims, labels=date_format("%d-%m-%Y"))
    
    sens_id <- sens_extract[id_loop]  
    fname_ts_t = paste(paste(paste('TimeSeries_total', sensor_stem, sep = ''), sens_id, sep = ''), '.png', sep='')
    fname_ts_r = paste(paste(paste('TimeSeries_residual', sensor_stem, sep = ''), sens_id, sep = ''), '.png', sep='')
    fname_ts_res_scat = paste(paste(paste('Residual_scatter', sensor_stem, sep = ''), sens_id, sep = ''), '.png', sep='')
    
    # p_ts_total <- p_ts_total + annotate(geom="text",as.POSIXct(strptime(annotate_xlox[sens_type], "%Y-%m-%d" )),
    #                                     y=unlist(annotate_yloc[sens_type])[1],label=paste("SPATIAL MAE = ",round(mae_spatial_, digits = 2)),fontface="bold")
    # p_ts_total <- p_ts_total + annotate(geom="text",as.POSIXct(strptime(annotate_xlox[sens_type], "%Y-%m-%d" )),
    #                                     y=unlist(annotate_yloc[sens_type])[2],label=paste("LALE MAE = ",round(mae_lale_, digits = 2)),fontface="bold")
    # p_ts_total <- p_ts_total + annotate(geom="text",as.POSIXct(strptime(annotate_xlox[sens_type], "%Y-%m-%d" )),
    #                                     y=unlist(annotate_yloc[sens_type])[3],label=paste("AMLP MAE = ",round(mae_amlp_, digits = 2)),fontface="bold")
    # 
    
    mae_lale = c(mae_lale, mean(abs(df_lale$output - df_lale$reconstruct_signal),na.rm=TRUE))
    mae_amlp = c(mae_amlp, mean(abs(df_lale$output  - df_amlp$reconstruct_signal),na.rm=TRUE))
    mae_spatial = c(mae_spatial, mean(abs(df_spat$output  - df_spat$reconstruct_signal),na.rm=TRUE))
    mae_cnn = c(mae_cnn, mean(abs(df_cnn$output  - df_cnn$reconstruct_signal),na.rm=TRUE))
    mae_lstm = c(mae_lstm, mean(abs(df_lale$output  - df_lale$reconstruct_signal),na.rm=TRUE))
    
    mape_lale = c(mape_lale, mean(abs( (df_lale$output - df_lale$reconstruct_signal)/df_lale$output*100), na.rm=TRUE))
    mape_amlp = c(mape_amlp, mean(abs( (df_amlp$output  - df_amlp$reconstruct_signal)/df_amlp$output*100), na.rm=TRUE))
    mape_spatial = c(mape_spatial, mean(abs( (df_spat$output  - df_spat$reconstruct_signal)/df_spat$output*100),na.rm=TRUE))
    mape_cnn = c(mape_cnn, mean(abs( (df_cnn$output  - df_cnn$reconstruct_signal)/df_cnn$output*100),na.rm=TRUE))
    mape_lstm = c(mape_lstm, mean(abs( (df_lstm$output  - df_lstm$reconstruct_signal)/df_lstm$output*100),na.rm=TRUE))
    
    
    ggsave(here(getwd(),results_dir, fname_ts_t),p_ts_total)

  } # end the loop over similar sensors
  if (write_stats){ # check whether to write stats
      
    mae_all_lale <- cbind( mean(mae_lale,na.rm=TRUE), 
                   sd(mae_lale,na.rm=TRUE),
                   mean(mape_lale,na.rm=TRUE))
    mae_all_amlp <- cbind( mean(mae_amlp,na.rm=TRUE), 
                           sd(mae_amlp,na.rm=TRUE),
                           mean(mape_amlp,na.rm=TRUE))
    mae_all_spat <- cbind( mean(mae_spatial,na.rm=TRUE), 
                           sd(mae_spatial,na.rm=TRUE),
                           mean(mape_spatial,na.rm=TRUE))
    mae_all_cnn <- cbind( mean(mae_cnn,na.rm=TRUE), 
                           sd(mae_cnn,na.rm=TRUE),
                          mean(mape_cnn,na.rm=TRUE))
    mae_all_lstm <- cbind( mean(mae_lstm,na.rm=TRUE), 
                           sd(mae_lstm,na.rm=TRUE),
                           mean(mape_lstm,na.rm=TRUE))
    
    
    if (sens_type == 1){
      col_names = c("sensor","AutoAI_MAE","AutoAI_SD","AutoAI_MAPE","AMLP_MAE", 
                    "AMLP_SD","AMLP_MAPE","CNN_MAE", "CNN_SD","CNN_MAPE",
                     "LSTM_MAE", "LSTM_SD","LSTM_MAPE",
                    "SPATIAL_MAE", "SPATIAL_SD","SPATIAL_MAPE")
      col_names_all = c("sensor","AutoAI","AMLP",
                    "CNN",  "LSTM",  "SPATIAL")
      append_file = FALSE
    }else{
      col_names = FALSE
      append_file = TRUE
      col_names_all = FALSE
    }
    
    write.table(cbind(lale_sensor_dir[sens_type], 
                     round(mae_all_lale,digits = 2), 
                      round(mae_all_amlp,digits = 2),
                      round(mae_all_cnn,digits = 2),
                      round(mae_all_lstm,digits = 2), 
                      round(mae_all_spat,digits = 2)),
                fout, append = append_file, 
                row.names = FALSE, 
                col.names =  col_names,
               # col.names = c("sensor", "AutoAI", "AMLP", "CNN","LSTM", "SPATIAL"), 
                sep = ',')
    
    write.table(cbind(lale_sensor_dir[sens_type],
                      round(mae_lale, digits = 2),
                      round(mae_amlp, digits = 2),
                      round(mae_cnn, digits = 2),
                      round(mae_lstm, digits = 2),
                      round(mae_spatial, digits = 2)),
                fout_mae, append = append_file, 
                col.names = col_names_all, row.names = FALSE, sep = ',')   
    
    
    write.table(cbind(lale_sensor_dir[sens_type],
                      round(mape_lale, digits = 2),
                      round(mape_amlp, digits = 2),
                      round(mape_cnn, digits = 2),
                      round(mape_lstm, digits = 2),
                      round(mape_spatial, digits = 2)),
                fout_mape, append = append_file, 
                col.names = col_names_all, row.names = FALSE, sep = ',')   
  } # end check on whether to write stats


  ### Finally let's look at the MAE reported for the pertinent sensors
  sensors_we_use = c(1,8,9)
  print(paste('The MAE at all Lale sensors is ', mean(mae_lale,na.rm=TRUE)))
  print(paste('The standard deviation of the Lale MAE is ', sd(mae_lale,na.rm=TRUE)))
  print(paste('The MAE at all AMLP sensors is ', mean(mae_amlp,na.rm=TRUE)))
  print(paste('The standard deviation of the AMLP MAE is ', sd(mae_amlp,na.rm=TRUE)))
  print(paste('The MAE at all SPATIAL sensors is ', mean(mae_spatial,na.rm=TRUE)))
  print(paste('The standard deviation of the AMLP MAE is ', sd(mae_spatial,na.rm=TRUE)))
  print(sens_type)
  
  lims_1 <- as.POSIXct(strptime(unlist(lims_arr[1]), format = "%Y-%m-%d"))
  lims_2 <- as.POSIXct(strptime(unlist(lims_arr[2]), format = "%Y-%m-%d"))
  lims_3 <- as.POSIXct(strptime(unlist(lims_arr[3]), format = "%Y-%m-%d"))
  
  ### Now let's save our figure objects for further formatting
  if (sens_type == 1){
    p_ts_total_temp <- p_ts_total + rremove("xlab") + 
      scale_x_datetime(limits =  lims_1, labels=date_format("%d-%m")) + 
      ylim(c(4,7))
  }
  
  if (sens_type == 2){
    p_ts_total_do <- p_ts_total + rremove("xlab")  + 
      scale_x_datetime(limits =  lims_2, labels=date_format("%d-%m")) 
  }
  
  if (sens_type == 3){
    p_ts_total_adcp <- p_ts_total + rremove("xlab") +
      scale_x_datetime(limits =  lims_3, labels=date_format("%d-%m"), 
                       breaks=date_breaks("4 day")) +ylim(c(0,30))
  }

}


if (create_grid_plot){
  g = ggarrange(p_ts_total_temp ,
                p_ts_total_do ,
                p_ts_total_adcp ,
                p_ts_res_temp   ,
                p_ts_res_do ,
                p_ts_res_adcp  ,
                nrow = 3, ncol=2, common.legend = T)
  
  g_review = ggarrange(p_ts_total_temp ,
                       p_ts_total_do ,
                       p_ts_total_adcp ,
                       nrow = 3, ncol=1, common.legend = T)
  
  ggsave(here(getwd(), results_dir, 'combined_images.png') ,g, height=4.8, width=6.4)
  ggsave(here(getwd(), results_dir, 'lstm_cnn_spatial.png') ,g_review, height=6.4, width=6.4)
  
}





