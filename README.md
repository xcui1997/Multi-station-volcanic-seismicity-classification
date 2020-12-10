# Multi -station volcanic seismicity classification

This is a brief introduction to the code. Here we use hierarchical clustering to classify the seismic events, and then regroup them
according to their peak frequency and frequency bandwidth.

## Requirement
The script require python and some commonly used packages.

## Data
The compressed package here include part of the data of 17 stations, which can be used to test the code after decompression.

## Run the script
You can run the test code as:
```
python Muti-station_volcanic_signals_classificaiton.py -P config_json
```
This will return the classification results.

The config_json shows the parameters you may need to modify and the files you need to perpare in advance.

The data is downloaded from iris and preprocessed (filter:1-15Hz, mean removal, instrument response correction, resample 100 Hz).
```


