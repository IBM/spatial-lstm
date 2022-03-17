### Process the SPATIAL experiments into same format as Lale/AMLP
library(here)
library(tidyverse)
library(ggpubr)
library(rstatix)
library("Hmisc")
library(GGally)
this.dir <- dirname(parent.frame(2)$ofile) # frame(3) also works.
setwd(this.dir)
print(getwd())
source('./util_fncs.R')

init_pwc_write = TRUE
create_grid_plot = FALSE

results_dir = './figures/'

## List the pertinent directories that store data, spatial results, amlp results, lale results...
# input data
raw_data_dir = '../sensor_data/interpolated_data/'
## Spatial data dir
dl_dir = '../deep_learning_experiments/prediction/'
# Lale data dire
lale_dir = '../machine_learning_experiments//autoAI_experiments//results/'
# AMLP dir
amlp_dir = '../machine_learning_experiments/amlp/results/'

## Load the raw and SPATIAL results.
### Files we use are (from Yihao)
sensors_to_use <- list(c(9), c(9), c(4)) # for temperature, oxy, and adcp respectively
sensor_stem_list <- c('Temp_Sensor_', 'Oxygen_Sensor_', 'ADCP_Sensor_')
spat_stem_list <- c('Diff_Temp_aqme_proc-mcnuttsisland_cage-19-',
                    'Diff_dissolved_oxygen_aqme_proc-mcnuttsisland_cage-19-',
                    'Diff_ADCP_sensor')

lale_sensor_dir = c('temperature/', 'oxygen/', 'adcp/')
amlp_sensor_dir = c('temp/', 'oxygen/', 'adcp/')


for (sens_type in seq(3, 3)){ # We just focus 
  if (sens_type == 3){
    ADCP = TRUE
    } else {
      ADCP = FALSE
    }
  sensor_stem <- sensor_stem_list[sens_type]
  sens_extract = unlist(sensors_to_use[sens_type])
  # 1) raw data files
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
  
  # 5) CNN data files
  cnn_fname <-  paste(paste(sensor_stem_list[sens_type], 
                            sens_extract, sep=''), '_CNN_prediction.csv', sep = '')
  
  # 6) LSTM data files
  lstm_fname <-  paste(paste(sensor_stem_list[sens_type], 
                             sens_extract, sep=''), '_LSTM_prediction.csv', sep = '')
  
  
  
  for (id_loop in seq(1, length(sens_extract))){
    sens_id <- sens_extract[id_loop]
    sens_id <- sens_extract[id_loop]
    sensor_fname = here(getwd(), raw_data_dir, raw_data_fname[id_loop])
    residual_fname = here(getwd(), lale_dir, lale_sensor_dir[sens_type], lale_fname[id_loop])
    amlp_fname_ = here(getwd(), amlp_dir, amlp_sensor_dir[sens_type], amlp_fname[id_loop])
    spat_fname_ = here(getwd(), dl_dir, spatial_fname[id_loop])
    print(sensor_fname)  
    print(residual_fname)
    if (sens_type == 3){ADCP_FLAG=TRUE}
    list_df = reconstruct_residual(sensor_fname, residual_fname, ADCP=ADCP_FLAG)
    list_df_amlp = reconstruct_residual_amlp(sensor_fname, amlp_fname_, ADCP=ADCP_FLAG)
    list_df_spat = reconstruct_residual_spat(sensor_fname, spat_fname_, ADCP=ADCP_FLAG)
    
    df_sensor = list_df[['df_sensor']]
    df_lale = list_df[['df_model']]
    df_amlp = list_df_amlp[['df_model']]
    df_spat = list_df_spat[['df_model']]
    if (sens_type == 1){ ## filter out crazy high value for temp
      df_lale$reconstruct_signal[df_lale$reconstruct_signal>8]  = NA
      df_lale$output[df_lale$output>8]  = NA
      df_amlp$reconstruct_signal[df_amlp$reconstruct_signal>8]  = NA
      df_amlp$output[df_amlp$output>8]  = NA
      df_spat$reconstruct_signal[df_spat$reconstruct_signal>8]  = NA
      df_spat$output[df_spat$output>8]  = NA
    }
    
    if (sens_type == 1){ ##and crazy low values
      lim_b <- 4.5
      date_beg_lale = as.POSIXct("2018-12-05","%Y-%m-%d", tz = "UTC") 
      df_spat <- df_spat %>% filter((date < date_beg_lale & 
                                       reconstruct_signal>lim_b)
                                    | date > date_beg_lale)
      
    }
    
    df <- combine_all_forecasts(df_lale, df_spat, df_amlp)
    df_model <- select(df, date,  observation, AutoAI, SPATIAL, AMLP)
    gdf <- df_model %>%
      gather(key = "dataset", value = "value", observation, AutoAI, SPATIAL, AMLP) %>%
      convert_as_factor(date, dataset)

    bxp <- ggboxplot(gdf, x = "dataset", y = "value", xlab = FALSE,
                     order = c("observation", "SPATIAL", "AutoAI", "AMLP"), 
                     color = "dataset", error.plot = "errorbar") +
      theme(panel.grid = element_blank(),
                        panel.background = element_blank(),
                         panel.border = element_rect(colour = "black", fill=NA, size=1),
                         axis.ticks.x = element_blank(), 
            axis.text.x = element_blank())
    
    data_rcorr <-as.matrix(df_model[, 2:5])

    ### Now let's save our figure objects for further formatting
    if (sens_type == 1){
      bxp_temp <- bxp
      r_temp <- rcorr(data_rcorr)
    }
    
    if (sens_type == 2){
      bxp_do <- bxp
      r_do <- rcorr(data_rcorr)
      
    }
    
    if (sens_type == 3){
      bxp_adcp <- bxp + labs(y = 'Speed (cm/s)') +
        ylim(c(0,20))
      r_adcp <- rcorr(data_rcorr)
      
    }

    
    # pairwise comparisons
    pwc <- gdf %>%
      t_test(value ~ dataset, paired = TRUE) %>%
      add_significance()
    pwc$sensor <-paste(sensor_stem, sens_id, sep = '')
    
    effect_size <- gdf  %>% cohens_d(value ~ dataset, paired = TRUE)
    pwc$effsize = effect_size$effsize
    pwc$magnitude = effect_size$magnitude
    
    if (init_pwc_write){
      init_pwc_write=FALSE
      write.table(pwc, here(getwd(), results_dir, 'pwc_res2.csv'),
                  append = FALSE,
                  sep = ",",
                  col.names = TRUE,
                  row.names = FALSE,
                  quote = FALSE)
    } else {
      write.table(pwc, here(getwd(), results_dir, 'pwc_res2.csv'),
                  append = TRUE,
                  sep = ",",
                  col.names = FALSE,
                  row.names = FALSE,
                  quote = FALSE)
    }
  }
  
}

if (create_grid_plot){
  g = ggarrange(bxp_temp +  labs(y= expression('Temperature ('*~degree*C*')')) ,
                bxp_do + labs(y = 'DO (mg/L)'),
                bxp_adcp + labs(y = 'Speed (cm/s)') +
                  ylim(c(0,20)),
                nrow = 1, ncol=3, common.legend = T)
  
  ggsave(here(getwd(), results_dir, 'boxplot_all.png') ,g, height=4.8, width=6.4)
  
}




ndays <- 7  # number of days to include for analysis
df_model <- df_model[1:(48*ndays),]

gdf <- df_model %>%
  gather(key = "variable", value = "value", observation, AutoAI, SPATIAL, AMLP) %>%
  convert_as_factor(date, variable)
head(gdf, 3)

gdf %>%
  group_by(date) %>%
  get_summary_stats(value, type = "mean_sd")


gdf %>%
  group_by(variable) %>%
  identify_outliers(value)



# pairwise comparisons
pwc <- gdf %>%
  t_test(value ~ variable, paired = TRUE) %>%
  add_significance()


#pwc_select <- pwc[pwc$group1=='observation' | pwc$group2=='observation',]
pwc$sensor <-paste(sensor_stem, sens_id, sep = '')

if (init_pwc_write){
  init_pwc_write=FALSE
  write.table(pwc, here(getwd(), results_dir, 'pwc_res2.csv'),
              append = FALSE,
              sep = ",",
              col.names = TRUE,
              row.names = FALSE,
              quote = FALSE)
} else {
  write.table(pwc, here(getwd(), results_dir, 'pwc_res2.csv'),
              append = TRUE,
              sep = ",",
              col.names = FALSE,
              row.names = FALSE,
              quote = FALSE)
}


