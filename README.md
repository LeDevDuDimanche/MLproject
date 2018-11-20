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

  

