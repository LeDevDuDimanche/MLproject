import os
import pdb
import json
from os.path import isfile, join
import re
from datetime import timedelta, datetime

#creates a subsample of the dataset in "../data" with the data from only the nb_websites most popular websites
#the interval of time we consider is (if d0 = first day of data collection)  [d0 , d0 + max_days_after]
def create_closed_world(nb_websites=20, max_days_after=10000, input_dir="../data/"):
	
	all_data_files = [f for f in os.listdir(input_dir) if isfile(join(input_dir, f))]
	file_name_pattern = r'(\d\d)-(\d\d)-(\d\d).*\.json'

	file_to_date = {}
	for f in all_data_files:
		match_obj = re.match(file_name_pattern, f, flags=0)
		if not match_obj:
			print("data file {0} doesn't match the expected file pattern".format(f))
			continue	

		day = match_obj.group(1)
		month = match_obj.group(2)
		year = "20"+match_obj.group(3) #cheating for using datetime. We need to specify we are in the 21st century

		file_date = datetime.strptime("{0} {1} {2}".format(day, month, year), "%d %m %Y")
		file_to_date.update({f: file_date})
		
	filename_with_min_date = min(file_to_date, key=file_to_date.get)
	min_date = file_to_date.get(filename_with_min_date)
	delta = timedelta(days=max_days_after)	
	min_plus_delta = min_date + delta 
	filtered = {f: time  for f, time in file_to_date.items() if time < min_plus_delta}
	#pdb.set_trace()
	print("number of websites kept", len(filtered))

	def get_pcap_index(pcap_str):
		return pcap_str[:-1*len(".pcap")]
	def is_in_closed_world(pcap_str):
		return int(get_pcap_index(pcap_str)) <= nb_websites - 1

	new_directory = "../data_cw{0}_day0_to_{1}/".format(nb_websites, max_days_after)
	if not os.path.exists(new_directory):
	    os.makedirs(new_directory)

	visited_websites_set = set([])
	for fname in filtered:
		print(fname)
		with open(input_dir + fname) as f:
			data_dict = json.loads(f.read())
			closed_world_dict = {k: v for k,v in data_dict.items() if is_in_closed_world(k) }
			visited_websites_set.update(set(closed_world_dict.keys()))
			with open(new_directory + fname, "w") as new_file:
				json.dump(closed_world_dict, new_file, indent=4)
							

	print("number of websites in dataset", len(visited_websites_set))
	print("visited websites: ", sorted(list(map(lambda x: int(get_pcap_index(x)), visited_websites_set))))
	
	#for each file f in this list of files keep recreate a file in the directory  
	#with the content of the file being only the object n.pcap where n <= nb_websites

if __name__ == '__main__':
	import sys
	create_closed_world(nb_websites=int(sys.argv[1]), max_days_after=int(sys.argv[2]))
