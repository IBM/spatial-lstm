# spatial-lstm
Repo related to our AAAI 2021 paper "A spatio-temporal LSTM model to forecast across multiple temporal and spatial scales".
Contact details:
 - Fearghal O'Donncha (feardonn@ie.ibm.com)
 - Yihao Hu ()

This repo includes three subdirectories
 - `sensor_data` sensor or observation data used in the study. For data ownership reasons we could only make ADCP data publicly available. Repo is structured to implement all experiments on that dataset. For those interested in a more detailed exploration of the work, please contact the authors to get access to temperature and oxygen data used in study
 - `machine_learning_experiments` - scripts to run two different machine learning experiments on the data. We use two popular AutoAI frameworks to generate the two implementations, namely [Lale](https://github.com/iBM/lale) and [AutoMLPipeline or AMLP](https://github.com/IBM/AutoMLPipeline.jl).
 - `deep_learning_experiments` scripts to run deep learning experinments used in the paper, namely CNN, LSTM, and bidirectional LSTM model that we term SPATIAL
 - `plots` R scripts to visualise and statistically analyse model results
