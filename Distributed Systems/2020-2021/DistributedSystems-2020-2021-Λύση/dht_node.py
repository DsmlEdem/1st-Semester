import hashlib,json,requests



""""
This Class does the necessary 
functions for each node.
"""
class Node:
	def __init__(self,ip,port,cord_ip=None,cord_port=None, replication_factor=1, consistency_type=None):
		m = hashlib.sha1()
		self.my_ip = str(ip)
		self.my_port = str(port)
		m.update(self.my_ip.encode('utf-8')+self.my_port.encode('utf-8'))
		self.id = int(m.hexdigest(),16)
		self.my_ip = self.my_ip
		self.my_port = self.my_port
		self.files = {}	# keys are str
		self.replication_factor = int(replication_factor)
		#Replica Manager Dictionary: {node, replicas_of_that_node_dict}
		self.replica_manager = {}	
		if(consistency_type == 'linearizability'):
			self.isLinear = True
		else:
			#if I type anything else it's eventual consistency.
			self.isLinear = False
		if cord_ip is None:
			self.cord_ip = self.my_ip
			self.cord_port = self.my_port
			self.nodes = []
			self.previous_node = {'id': self.id, 'ip' : self.my_ip, 'port' : self.my_port}
			self.next_node = {'id': self.id, 'ip' : self.my_ip, 'port' : self.my_port}
		else:
			self.cord_ip = cord_ip
			self.cord_port = cord_port
			self.next_previous_nodes()

	"""
	HELPER FUNCTIONS -  GETTERS AND SETTERS.
	"""
	def is_coord(self):
		return self.my_ip == self.cord_ip and self.my_port == self.cord_port
	
	def get_coord_IP(self):
		return self.cord_ip

	def get_coord_port(self):
		return self.cord_port

	def get_IP(self):
		return self.my_ip

	def get_port(self):
		return self.my_port

	def get_ID(self):
		return self.id

	def get_previous_node(self):
		return self.previous_node

	def set_previous_node(self, new_pred):
		self.previous_node = new_pred

	def get_next_node(self):
		return self.next_node
	
	def set_next_node(self, new_suc):
		self.next_node = new_suc

	def insert_file(self, key, unhashed_key, value):
		self.files[str(key)] = (unhashed_key, value)

	def contains_file(self, key):
		return str(key) in self.files

	def delete_file(self, key):
		del self.files[str(key)]
		return

	def get_replication_factor(self):
		return self.replication_factor

	def key_is_mine(self, key):
		if(self.previous_node['id'] == self.next_node['id'] and self.previous_node['id'] == self.get_ID()):
			return True
		if(int(self.previous_node['id'])>int(self.id) and (int(key) > int(self.previous_node['id']) or int(key) < int(self.id))):
			return True
		return (int(key)>int(self.previous_node['id']) and int(key) <= int(self.id))

	def get_files(self):
		return self.files
	
	def get_replica_manager(self):
		return self.replica_manager

	def dict_append(self, dict):
		self.files.update(dict)

	def add_node_to_RM(self, node_id, dict):
		self.replica_manager[int(node_id)] = dict
	
	def add_file_to_RM(self, node_id, key, unhashed_key, value):
		self.replica_manager[int(node_id)][str(key)] = (unhashed_key, value)
	
	def delete_node_from_RM(self, id):
		try:
			self.replica_manager.pop(int(id))
		except KeyError: 
			return
		
	def node_in_RM(self, id):
		return int(id) in self.replica_manager.keys()
	
	def file_in_RM(self, file_key):
		for node_id, replica in self.replica_manager.items():
			if(file_key in replica.keys()):
				return node_id
		return -1
	
	def get_file_from_RM(self, original_node_id, key):
		if int(original_node_id) in self.replica_manager.keys():
			if str(key) in self.replica_manager[int(original_node_id)].keys():
				return self.replica_manager[int(original_node_id)][str(key)][1]
		return None

	
	def delete_file_from_RM(self, id, key):
		if int(id) in self.replica_manager.keys():
			temp_dict = self.replica_manager[int(id)]
			temp_dict.pop(str(key))
			self.replica_manager[int(id)] = temp_dict
	def print(self):
		return 'My IP'.encode("utf-8")+self.my_ip+'and port'.encode("utf-8")+self.my_port +'and hashed ID'.encode("utf-8")+str(self.id).encode('utf-8')

	def announce_departure(self):
		response = requests.get(url = "http://"+self.cord_ip+":"+self.cord_port+"/node_departure", params = {'id':self.id, 'ip':self.my_ip, 'port':self.my_port})
		json_data = json.loads(response.text)
		print(json_data)

	def get_replica_sources(self):
		response = requests.get(url = "http://"+self.cord_ip+":"+self.cord_port+"/replica_nodes", params = {'id':self.id, 'ip':self.my_ip, 'port':self.my_port})
		json_data = json.loads(response.text)

		#read list with replication sources
		source_list = json_data['replica_sources'] #replica_sources_cell : {id, ip, port} : list
		for node_info_list in source_list:
			print(node_info_list)
			#ask current_node for files
			current_node_id = node_info_list[0]
			current_node_ip = node_info_list[1]
			current_node_port = node_info_list[2]
			response = requests.get(url = "http://"+current_node_ip+":"+current_node_port+"/node_files", params = {})
			json_data = json.loads(response.text)

			replica_dict = json_data['files']
			self.replica_manager[int(current_node_id)] = replica_dict

	# Ask coordinator for successors responsibles for my replicas and send my files to them
	def get_replicas_keeper(self):
		response = requests.get(url = "http://"+self.cord_ip+":"+self.cord_port+"/replica_keepers", params = {'id':self.id, 'ip':self.my_ip, 'port':self.my_port})
		json_data = json.loads(response.text)

		#read list with replication sources
		keepers_list = json_data['replica_keepers'] #replica_sources_cell : {id, ip, port} : list
		for node_info_list in keepers_list:
			print("list of keepers")
			print(node_info_list)
			#send files to current node
			current_node_id = node_info_list[0]
			current_node_ip = node_info_list[1]
			current_node_port = node_info_list[2]
			response = requests.post(url = "http://"+current_node_ip+":"+current_node_port+"/accept_replicas", params = {'id':self.id, 'files':json.dumps(self.files)})
			print(response)

	def next_previous_nodes(self):
		response = requests.get(url = "http://"+self.cord_ip+":"+self.cord_port+"/accept_node", params = {'id':self.id, 'ip':self.my_ip, 'port':self.my_port})
		json_data = json.loads(response.text)

		### Read previous node
		id_pred = json_data['previous_node_id']
		ip_pred = json_data['previous_node_ip']
		port_pred = json_data['previous_node_port']
		self.previous_node = {'id': id_pred, 'ip' : ip_pred, 'port' : port_pred}
	
		### Read next node
		id_suc = json_data['next_node_id']
		ip_suc = json_data['next_node_ip']
		port_suc = json_data['next_node_port']
		self.next_node = {'id': id_suc, 'ip' : ip_suc, 'port' : port_suc}

		print("My previous node is: {}:{} and the next one is: {}:{}".format(self.previous_node['ip'], self.previous_node['port'], self.next_node['ip'], self.next_node['port']))

