# CS433_project
Code for CS433 Project 2

### Structure

**code/** - Scripts for data processing and analysis

- classify_rnn.py: Sample code using RNN for classification
- process_trace_tshark.py: File to process raw PCAP file and obtain the JSON files in data/	

**data/** - Processed data files for analysis

- Files are of the format DD-MM-YY_tag.json or DD-MM-YY_tag_2.json. tag is just a number to indicate which VM the experiment was run on. Only the date is relevant for analysis.

- Each JSON file has keys of the format **n.pcap**, where **n** indicates the index of a particular webpage in the list **short_list_500**. There are 1500 webpages, so keys will be from 0.pcap to 1499.pcap. Note that all files might not have data for all 1500 webpages. 

- For each webpage, there are three values. **sent** is the sequence of sent pcaket sizes. **received** is the sequence of received packet sizes. **order** is the sequence in which packets where sent or received — -1 indicates a received packet and 1 indicates a sent packet.

  
###How to run

- Run everything with Python 2 as we developed with this version (2.7) and cannot guarantee compilation with Python 3.0 or above.

- Install the following dependencies :
	-numpy 1.15.4
	-sklearn 0.0
	-keras 2.2.4
	-tensorflow 1.12
	-hyperopt 0.1.1

- Run the create_closed_world script in the /code/ directory. It will create a data directory with the filtered json files. You can modify the number of categories in the closed world as well as the number of days to filter. Default is 20, 10 000 (the whole time period)

- Create a 'results' and a 'logs' directories at the same level as data/ and code/. The NN scripts will save the accuracies and hyperparameters there.

- In classify_rnn.py and classify_cnn.py, change the NUM_CLASSES and NUM_DAYS global variables by the parameters you changed in create_closed_world, if you did change them. Default are 20, 10 000.

- Just run them like it's hot :-)
