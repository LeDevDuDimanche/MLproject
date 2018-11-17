import sys
import subprocess
import os
import json

OPTIONS = [
		'tcp.desegment_tcp_streams:FALSE',
		'ssl.desegment_ssl_records:FALSE',
		'ssl.desegment_ssl_application_data:FALSE',
]
TSHARK_CMD = 'tshark -r {filepath} -E header=y -E separator=; -n -T fields {fields} {options}'
FIELDS = [
					# Frame
					'frame.number',
					'frame.time',
					'frame.time_delta',

					# IP
					'ip.src',
					'ip.dst',
					'ip.proto',
					'ip.len',
					'ip.hdr_len',

					# UDP
					'udp.srcport',
					'udp.dstport',
					'udp.length',

					# UDP wireshark analysis
					'udp.stream',

					# TCP
					'tcp.srcport',
					'tcp.dstport',
					'tcp.seq',
					'tcp.flags.ack',
					'tcp.flags.syn',
					'tcp.flags.fin',
					'tcp.len',
					'tcp.hdr_len',

					# TCP wireshark analysis
					'tcp.stream',
					'tcp.analysis',
					'tcp.analysis.retransmission',

					# DNS
					'dns.qry.name',
					'dns.cname',
					'dns.a',

					# TLS
					'ssl.handshake.extensions_server_name',
					'ssl.handshake.ciphersuite',
					'ssl.handshake.length',
					'ssl.heartbeat_message',
					'ssl.record.content_type',
					'ssl.record.length',

					# wireshark columns
					'_ws.col.Protocol',
				 ]

# format fields
FIELD_LABEL = ' -e '
FIELDS_FORMATTED = FIELD_LABEL
FIELDS_FORMATTED += FIELD_LABEL.join(FIELDS)

# format options
OPTION_LABEL = ' -o'
OPTIONS_FORMATTED = OPTION_LABEL
OPTIONS_FORMATTED += OPTION_LABEL.join(OPTIONS)


def process_packets(dirname, output):
	print dirname
	data_dict = {}
	flist = os.listdir(dirname)
	#print flist[:2]
	#flist = [x for x in flist if x == "680.pcap"]
	print len(flist)
	for fname in flist:
		#print fname
		#dname = '\"/Volumes/NO NAME/01-06-18_long/\"'
		pcap_fpath = dirname + fname
		sent_lengths = []
		received_lengths = []
		order = []
		flag = 0

		tshark_cmd = TSHARK_CMD.format(filepath=pcap_fpath,
																	 fields=FIELDS_FORMATTED,
																	 options=OPTIONS_FORMATTED)
		try:
			tshark_out = subprocess.check_output(tshark_cmd.split())
			IP_ADDR = ['1.1.1.1', '1.0.0.1']
			data_dict[fname] = {}
			for pkt in tshark_out.split('\n'):
					fields = pkt.split(';')
					print fields
					proto = fields[-1]
					if 'TLS' in proto or 'SSL' in proto:
							tls_content_type = fields[-3]
							if tls_content_type == '':
									# TODO: this is likely to be a "ignored unknown record" in
									# wireshark
									continue
							if '23' not in tls_content_type:  # not application data
									continue
							heartbeat = fields[-4]
							if heartbeat != '':  # heartbeat found
									continue
							# if flag == 0:
							# 	continue
							#print fields[0]
							tls_record_length = map(int, fields[-2].split(','))
							#print tls_record_length
							if fields[12] != '' and fields[13] != '' and fields[3] != '' and fields[4] != '':

								sport, dport = int(fields[12]), int(fields[13])
								src, dst = fields[3], fields[4]
								if  sport == 443 and src in IP_ADDR:
										#print 'here'
										#if flag = 0:
										received_lengths += tls_record_length
										order += [-1] * len(tls_record_length)
								elif dport == 443 and dst in IP_ADDR:
										flag = 1
										sent_lengths += tls_record_length
										order += [1] * len(tls_record_length)
								#else:
								#raise Exception("TLS packet to port: %s" % dport)
								#break
			#print received_lengths
			data_dict[fname]['sent'] = sent_lengths
			data_dict[fname]['received'] = received_lengths
			data_dict[fname]['order'] = order
			
			if len(data_dict) % 1000 == 0:
				print "Done", len(data_dict)
				with open(output, 'w') as f:
					print >>f, json.dumps(data_dict, indent = 4)
			#print received_lengths
		except Exception as e:
			print e


	# 	with open('lengths_FALSE', 'a') as f2:
	# 		print >>f2, fname, '_'.join([str(x) for x in received_lengths])

	with open(output, 'w') as f:
		print >>f, json.dumps(data_dict, indent = 4)

def compare():
	dataf = []
	datat = []
	flist = ['lengths_FALSE', 'lengths_TRUE']
	ct = 0
	for fname in flist:
		print fname
		with open(fname) as f:
			lines = f.readlines()
			for line in lines:
				a = line.strip().split()
				if ct == 0:
					dataf.append(a[-1])
				else:
					datat.append(a[-1])
		ct += 1
	for i in range(0, len(datat)):
		if datat[i] != dataf[i]:
			print i, dataf, datat

if __name__ == "__main__":
	
	dirnames = ["04-09-18"]
	for dirname in dirnames:
		output = "04-09-18_11.json"
		process_packets("/Volumes/WDPassport/pcaps/11/" + dirname + "/", output)
