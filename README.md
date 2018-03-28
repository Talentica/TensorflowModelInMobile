# Tensorflow Model In Mobile
This project demonstrate how we can generate tensorflow model and use that model in mobile. Here the model is motion detector which make use of linear acceleration sensor data and apply IIR filter on raw data, then using threshold value to detect whether user is moving or not. The paramters for IIR filter and threshold is part of model. This project include accompanying android as well ios app which demonstrate how we can make use of model in mobile using tensorflow-mobile API.
Model is written in python and is exported & optimized for mobile at backend. The final app looks as follows

## Android App
![alt text](https://drive.google.com/uc?export=view&id=1smEkDCAN86y4CYhwoFFwcJZVq1bgFEuV)
## iOS App
![alt text](https://drive.google.com/uc?export=view&id=1vNAKYo6b2P_EsZ_3XnO3wjYFMxe9ibaI)

# Run locally
* docker-compose up -d. This image contains all necessary packages required to build tensorflow application. It also include cloud9 IDE and jupyter for easy development.
* Open cloud9 IDE by visiting at http://localhost
* change directory to /notebook
* Run jupyter by running command : /run_jupyter.sh --allow-root
* The above command will print url, open that url to access jupyter. 
* The notebook demonstrate how we can export the model ( or change the paramters if required)

