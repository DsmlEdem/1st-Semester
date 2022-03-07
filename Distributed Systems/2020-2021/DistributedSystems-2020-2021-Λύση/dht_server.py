""""


ToyChord Implementation using Flask Rest API 
Distributed Systems 2020-2021

MSc Data Science and Machine Learning
National Technical University of Athens


"""


#--------- Import Libraries ---------$
import hashlib,flask,json,threading,time,signal,sys,psutil,os,requests,bisect
from flask import Flask, make_response, jsonify, Response
from flask import request
from flask_script import Manager, Server
import dht_node as dn #import node.py
app = Flask(__name__)


#--------- Instantiate basic variables ---------$
#ring_nodes is a dictionary with key: ID of node and values: IP and Port of said node.
ring_nodes = {}
#ring_ids is a list containing all the IDs of all the nodes.
ring_ids = []
#current_node is an instance of the node running this code.
current_node = None
#app_thread is a thread. each node has 2 threads, one for requests and one starting the session.
app_thread = None


#First Testing end-point of our rest API.
@app.route('/')
def hello():
    return "Hello World!"


"""
-------------------------------------------------------------------------------
BASIC FUNCTIONS OF THE 
DISTRIBUTED HASH TABLE
-------------------------------------------------------------------------------
"""
@app.route('/insert', methods=['GET', 'POST'])
def insert_new_file():
	"""
	This function handles insert of files, 
	either on this node or 
	forwarding to the next node until 
	it finds the corresponding node.
	"""
	#get key from request, and produce its hash.
	unhashed_key = request.args.get('key')
	#get value from request
	value = request.args.get('value')
	#get ip from request
	client_ip = request.args.get('client_ip')
	#get port from request
	client_port = request.args.get('client_port')
	# print message for sanity check
	print('Got new file insert request from client with ip:{} and port:{}'.format(client_ip,client_port))

	### Hash function SHA1 ###
	m = hashlib.sha1()
	m.update(unhashed_key.encode('utf-8'))
	key = int(m.hexdigest(),16)
	###
	message = ''
	#first case: the key belongs to me
	if(current_node.key_is_mine(key)):
		if(current_node.replication_factor == 1 or (not current_node.isLinear)):
			### Store file in this node
			current_node.insert_file(key, unhashed_key, value)
			message = 'Stored file'

			#Answer to cli for successful insert
			cli_message = 'File stored successfully'
			cli_response = requests.post(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'message':		message})

			### If replication_factor>1 --> eventual consistency --> propagate to next replica manager
			if(current_node.replication_factor >1):
				#Propagate insert to successors nodes with my replicas
				response = requests.get(url = "http://"+str(current_node.get_next_node()['ip'])+":"+str(current_node.get_next_node()['port'])+"/insert_propagation_eventual_consistency", \
					params = {'key':unhashed_key, 'value':value, 'origin_id':current_node.get_ID()})

		else:
			current_node.insert_file(key, unhashed_key, value)
			### Propagate write of file and last node answers with success message
			response = requests.get(url = "http://"+str(current_node.get_next_node()['ip'])+":"+str(current_node.get_next_node()['port'])+"/insert_propagation", \
				params = {'key':unhashed_key, 'value':value, 'origin_id':current_node.get_ID(), 'client_ip':client_ip, 'client_port':client_port})
			return jsonify(
				status='Propagating file to chain of replicas'
			)
	else:
		### second case: the key doesn't belong to me, pass the request to the next node.
		next_node_ip = current_node.get_next_node()['ip']
		next_node_port = current_node.get_next_node()['port']
		response = requests.get(url = "http://"+str(next_node_ip)+":"+str(next_node_port)+"/insert", \
			params = {'key':unhashed_key, 'value':value, 'client_ip':client_ip, 'client_port':client_port})
		json_data = json.loads(response.text)
		print(json_data)
		message = 'forward the insert query to the next node: {}:{}'.format(next_node_ip,next_node_port)

	return jsonify(
			status=message
	)

@app.route('/query', methods=['GET', 'POST'])
def query_file():
	"""
	This function handles queries of files.
	"""
	#as usual, request key, ip,port etc.
	unhashed_key = request.args.get('key')
	client_ip = request.args.get('client_ip')
	client_port = request.args.get('client_port')
	#this is the first node the query was placed
	first_node_id = request.args.get('first_node_id')
	direct_reply = True
	message = ''
	r_value = ''
	if(first_node_id is None):
		first_node_id = current_node.get_ID()
	elif(str(first_node_id)==str(current_node.get_ID())):
		return jsonify(
			status="ok",
			value = message
		)
	if(client_ip is None or client_port is None):
		client_ip = request.environ['REMOTE_ADDR']
		client_port = request.environ['REMOTE_PORT']
		direct_reply = False
	if(unhashed_key == '*'):
		message = 'You triggered the star. RELEASE THE KRAKEN'
		response = requests.post(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'node_id':current_node.get_ID(), 'files':json.dumps(current_node.get_files())})
		next_node_ip = current_node.get_next_node()['ip']
		next_node_port = current_node.get_next_node()['port']
		response = requests.get(url = "http://"+str(next_node_ip)+":"+str(next_node_port)+"/query", params = {'key':unhashed_key, 'client_ip':client_ip, 'client_port':client_port, 'first_node_id':first_node_id})
	else:
		### Hash function ###
		m = hashlib.sha1()
		m.update(unhashed_key.encode('utf-8'))
		key = int(m.hexdigest(),16)
		key = str(key)

		original_node = current_node.file_in_RM(key)
		if(current_node.key_is_mine(key)):
			# Replication_factor = 1 means no replication
			# or eventual consistency
			if(current_node.replication_factor == 1 or (not current_node.isLinear)): 
				### Search file in this node
				if(current_node.contains_file(key)):
					message = 'File found succesfully!'
					r_value = current_node.get_files()[str(key)][1]
				else:
					message = 'File not found!'
					r_value = 'None'
				response = requests.get(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'key':unhashed_key, 'value':r_value})

			else:
				### Propagating query to last replica manager responsible for my files
				response = requests.get(url = "http://"+str(current_node.get_next_node()['ip'])+":"+str(current_node.get_next_node()['port'])+"/query_propagation", \
					params = {'key':unhashed_key, 'origin_id':current_node.get_ID(), 'client_ip':client_ip, 'client_port':client_port})
				return jsonify(
					status='Propagating query to chain of replicas'
				)
		elif original_node>=0:
			if not current_node.isLinear:
			### Reply to CLI with value stored in dedicated replica
				file_value = current_node.get_file_from_RM(original_node, key)
				if not file_value is None:
					message = 'File found succesfully!'
					r_value = file_value
				else:
					message = 'File not found!'
					r_value = 'None'
				response = requests.get(url = "http://"+str(client_ip)+":"+str(client_port)+"/",
					params = {'key':unhashed_key, 'value':r_value, 'message':message})

			else:
				response = requests.get(url = "http://"+str(client_ip)+":"+str(client_port)+"/",params = {'key':unhashed_key, 'value':r_value, 'message':message})

				json_data = json.loads(response.text)
				if(json_data['message'] == 'Replica not found'):
					#this means i'm the last RM for original node -> return key-value pair
					file_value = current_node.get_file_from_RM(original_node,key)
					if not file_value is None:
						message = 'File found succesfully!'
						r_value = file_value
					else:
						message = 'File not found!'
						r_value = 'None'
					response = requests.get(url="http://" + str(client_ip) + ":" + str(client_port) + "/", \
											params={'key': unhashed_key, 'value': r_value, 'message': message})
				return jsonify(
						status='Propagating query to chain of replicas'
					)


		else:
			### Pass request to next node
			next_node_ip = current_node.get_next_node()['ip']
			next_node_port = current_node.get_next_node()['port']
			response = requests.get(url = "http://"+str(next_node_ip)+":"+str(next_node_port)+"/query", params = {'key':unhashed_key, 'client_ip':client_ip, 'client_port':client_port})
			json_data = json.loads(response.text)
			print(json_data)
			message = 'forwarding query request to my next_node: {}:{}'.format(next_node_ip,next_node_port)

	return jsonify(
			status="ok",
			value = message
	)
@app.route('/delete', methods=['GET', 'POST'])
def delete_thisfile():
	"""
	This function handles
	file deletion
	BOTH in nodes
	and in
	replicas.
	"""
	#as usual, request key, ip,port etc.
	unhashed_key = request.args.get('key')
	client_ip = request.args.get('client_ip')
	client_port = request.args.get('client_port')
	### Hash function ###
	m = hashlib.sha1()
	m.update(unhashed_key.encode('utf-8'))
	key = int(m.hexdigest(),16)
	key = str(key)

	message = ''

	if(current_node.key_is_mine(key)):
		if(key in current_node.get_files().keys()):
			# Replication_factor = 1 means no replication
			# or eventual consistency
			if(current_node.replication_factor == 1 or (not current_node.isLinear)):
				current_node.delete_file(key)
				message = 'Deleted the file'

				### Respond to CLI
				response = requests.post(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'message':"Success"})

				if(current_node.replication_factor > 1): #eventual consistency
					### Propagate delete to next_nodes with my replica
					response = requests.get(url = "http://"+str(current_node.get_next_node()['ip'])+":"+str(current_node.get_next_node()['port'])+"/delete_propagation_eventual_consistency", \
						params = {'key':unhashed_key, 'origin_id':current_node.get_ID()})


			else:	#Chain replication
				current_node.delete_file(key)
				### Propagate write of file and last node answers with success message
				response = requests.get(url = "http://"+str(current_node.get_next_node()['ip'])+":"+str(current_node.get_next_node()['port'])+"/delete_propagation", \
					params = {'key':unhashed_key, 'origin_id':current_node.get_ID(), 'client_ip':client_ip, 'client_port':client_port})
				return jsonify(
					status='Propagating deletion to chain of replicas'
				)
		else:
			message = 'File not found'
		
	else:
		### Pass request to next node
		next_node_ip = current_node.get_next_node()['ip']
		next_node_port = current_node.get_next_node()['port']
		response = requests.get(url = "http://"+str(next_node_ip)+":"+str(next_node_port)+"/delete", params = {'key':unhashed_key, 'client_ip':client_ip, 'client_port':client_port})
		json_data = json.loads(response.text)
		print(json_data)
		message = 'send the delete query to the next node: {}:{}'.format(next_node_ip,next_node_port)

	return jsonify(
			status=message
	)
@app.route('/topology', methods=['GET', 'POST'])
def topology():
	"""
	This function
	prints the topology 
	of the ring.
	"""
	if(current_node.is_coord()):
		json_reply = {}
		for i in range(len(ring_ids)):
			curr_id = ring_ids[i]
			curr_ip = ring_nodes[curr_id]['ip']
			curr_port = ring_nodes[curr_id]['port']
			json_reply['node-'+str(i)] = 'ID: '+ str(curr_id)+ ' -- IP:'+ str(curr_ip) + ' -- PORT: '+ str(curr_port)

		return jsonify(

			message = "Connected nodes: " + str(len(ring_ids)),
			nodes = json_reply
		)
	else:
		return jsonify(
			message = "I am not the coordinator",
		)
@app.route('/node_files', methods=['GET', 'POST'])
def node_files():
	"""
	This function returns
	a json object
	with all the files of the node.
	"""
	all_files = current_node.get_files()
	return jsonify(
		message = "This node has the following files:" + str(len(all_files)),
		files = all_files
	)
@app.route('/node_replicas', methods=['GET', 'POST'])
def node_replicas():
	"""
	This function returns
	a json object
	with all the replicas of the node.
	"""
	all_files = current_node.get_replica_manager()
	return jsonify(
		message = "All replicas kept from other nodes of this node : " + str(len(all_files)),
		files = all_files
	)
	
@app.route('/accept_new_node', methods=['GET', 'POST'])
def accept_new_node():
	"""
	When a new node enters the ring,
	has to hit this endpoint,
	to inform the rest of the ring 
	that he is ready
	to accept files
	from the next node.
	"""
	if current_node.is_coord():
		id_node = request.args.get('id')
		id_node = int(id_node)
		ip_node = request.args.get('ip')
		port_node = request.args.get('port')

		#Find next_node
		current_index = ring_ids.index(id_node)

		index_next_node = current_index + 1
		if(index_next_node == len(ring_ids)):
			index_next_node = 0
		
		ip_suc = ring_nodes[ring_ids[index_next_node]]['ip']
		port_suc = ring_nodes[ring_ids[index_next_node]]['port']

		#Find previous_node of new node (to pass as old_previous_node_id to the next_node)
		old_previous_node_id = current_index-1
		if(old_previous_node_id == -1):
			old_previous_node_id = len(ring_ids)-1
	
		###Notify next_node to send keys to predeccessor (who is the new node)
		response = requests.get(url = "http://"+ip_suc+":"+port_suc+"/send_keys_to_previous_node", params = {'id':id_node, 'ip':ip_node, 'port':port_node, 'old_previous_node_id': old_previous_node_id})
		print(response.content)
		json_data = json.loads(response.text)
		if(json_data['status'] == "Success"):
			#Tell new node that he is ready to send replicas of himself
			response = requests.get(url = "http://"+ip_node+":"+port_node+"/confirmation_to_send_replicas", params = {})


		return jsonify(
	    	answer="Success"
	    	#id=id_node
		)
	
	else:
		return jsonify(
	    answer="Fail, I am not the coordinator",
	    #id=id_node
		)

@app.route('/accept_files', methods=['POST'])
def accept_files():
	"""
	When a new node enters the ring, 
	he gets files from the next node,
	this function does that.
	"""
	files = request.args.get('files')
	json_dict = json.loads(files)
	message = 'success'
	print("Got my keys from:{}:{}".format(request.environ['REMOTE_ADDR'],request.environ['REMOTE_PORT']))

	current_node.dict_append(json_dict)

	return jsonify(
			status=message
	)
@app.route('/accept_node', methods=['GET', 'POST'])
def accept_node():
	#check if the node is the coordinator
	if current_node.is_coord():
		id_node = request.args.get('id')
		id_node = int(id_node)
		ip_node = request.args.get('ip')
		port_node = request.args.get('port')
		#Put new node in node list
		ring_nodes[id_node] = {'ip' : ip_node, 'port' : port_node}
		bisect.insort(ring_ids, id_node)	# Keep id in separate list
		
		current_index = ring_ids.index(id_node)
		### find previous node
		index_previous_node = current_index-1
		if(index_previous_node == -1):
			index_previous_node = len(ring_ids)-1
		
		ip_pred = ring_nodes[ring_ids[index_previous_node]]['ip']
		port_pred = ring_nodes[ring_ids[index_previous_node]]['port']

		### Update the next node of previous node
		response = requests.get(url = "http://"+str(ip_pred)+":"+str(port_pred)+"/update_next_node", params = {'id':id_node, 'ip':ip_node, 'port':port_node})

		### find next node
		index_next_node = current_index + 1
		if(index_next_node == len(ring_ids)):
			index_next_node = 0

		ip_suc = ring_nodes[ring_ids[index_next_node]]['ip']
		port_suc = ring_nodes[ring_ids[index_next_node]]['port']

		### Update the previous node of the next node 
		response = requests.get(url = "http://"+ip_suc+":"+port_suc+"/update_previous_node", params = {'id':id_node, 'ip':ip_node, 'port':port_node})
		""" 
			when a new node enters the ring, he inherits some keys from the next node.
			If the next node's next node keeps replicas of the next node, he has to 
			remove the keys that went to the new node.
		"""
		current_index = ring_ids.index(id_node)
		pending_next_nodes = current_node.replication_factor - 1
		current_next_node = current_index
		while(pending_next_nodes>0):
			current_next_node = current_next_node + 1
			if(current_next_node == len(ring_ids)):
				current_next_node = 0
			
			### Find k-(previous_nodeIP)
			current_previous_node = current_next_node
			pending_previous_nodes = current_node.replication_factor
			while(pending_previous_nodes>0):
				current_previous_node -= 1
				if(current_previous_node == -1):
					current_previous_node = len(ring_ids)-1
				
				# pair(current_next_node forget about current_previous_node)
				cur_next_node_ip = ring_nodes[ring_ids[current_next_node]]['ip']
				cur_next_node_port = ring_nodes[ring_ids[current_next_node]]['port']

				if(ip_node != cur_next_node_ip):
					response = requests.get(url = "http://"+cur_next_node_ip+":"+cur_next_node_port+"/replicas_deletion", params = {'id':ring_ids[current_previous_node]})

					json_data = json.loads(response.text)
					if(json_data['status'] == "Success"):
						print("Informed {} to delete replicas of node: {}".format(current_next_node, ring_ids[current_previous_node]))

				#Update pending previous_nodes
				pending_previous_nodes -= 1
				
			pending_next_nodes -=1
		### 

		return jsonify(
        status="Success",
        previous_node_id = str(ring_ids[index_previous_node]),
		previous_node_ip = str(ip_pred),
		previous_node_port = str(port_pred),

		next_node_id = str(ring_ids[index_next_node]),
		next_node_ip = str(ip_suc),
		next_node_port = str(port_suc)
    )
	else:
		return jsonify(
        answer="Fail",
        #id=id_node
    )

@app.route('/accept_replicas', methods=['POST'])
def accept_replicas():
	"""
	This function,
	similar to accept_files,
	when a node enters the ring,
	he gets replicas (and files).
	So, we handle replicas distribution.
	"""
	id = request.args.get('id')
	files = request.args.get('files')
	json_dict = json.loads(files)
	message = 'success'
	print("Got replica keys from:{}:{}".format(request.environ['REMOTE_ADDR'],request.environ['REMOTE_PORT']))

	current_node.add_node_to_RM(int(id),json_dict)


	return jsonify(
			status=message
	)


@app.route('/update_previous_node', methods=['GET', 'POST'])
def update_previous_node():
	"""
	When a new node enters the ring,
	we have to renew every node's 
	neighbors, this function
	lets a node know who is his
	previous node.
	"""
	id = request.args.get('id')
	ip = request.args.get('ip')
	port = request.args.get('port')
	pred = {'id' : id, 'ip' : ip, 'port': port}
	old_previous_node_id = current_node.get_previous_node()['id']
	
	current_node.set_previous_node(pred)
	
	print("Just updated my previous_node with id:{}, ip:{}, port:{}".format(id,ip,port))

	### Completed previous_node update
	return jsonify(
			status="Success"
	)

@app.route('/update_next_node', methods=['GET', 'POST'])
def update_next_node():
	"""
	When a new node enters the ring,
	we have to renew every node's 
	neighbors, this function
	lets a node know who is his
	next node.
	"""
	id = request.args.get('id')
	ip = request.args.get('ip')
	port = request.args.get('port')
	print("Request to update next_node")
	suc = {'id' : id, 'ip' : ip, 'port': port}
	current_node.set_next_node(suc)
	return jsonify(
			status="Success"
		)
@app.route('/send_keys_to_previous_node', methods=['GET', 'POST'])
def send_keys_to_previous_node():
	"""
	When a node departs,
	he has to send his keys
	to the previous node.
	"""
	id = request.args.get('id')
	ip = request.args.get('ip')
	port = request.args.get('port')
	pred = {'id' : id, 'ip' : ip, 'port': port}

	new_previous_node_id = id
	# if i'm the last node (coordinator), i don't need to send my files anywhere.
	if(current_node.get_ID() != new_previous_node_id):	
		###Send appropriate files to previous_node
		#Compute dictionary to send to previous_node
		transfer_dict = {}
		mark_to_remove = []
		
		for key, value in current_node.get_files().items():
			if(not current_node.key_is_mine(key)):
				transfer_dict[key] = value
				mark_to_remove.append(key)

		#Remove files from this node
		for key in mark_to_remove:
			current_node.delete_file(key)

		#Send dictionary with http POST request
		response = requests.post(url = "http://"+str(ip)+":"+str(port)+"/accept_files", params = {'files':json.dumps(transfer_dict)})

		### Notify coordinator to remove keys  - from the nodes that keep my replicas - that I just send to previous_node from my dedicated replica
		response = requests.post(url = "http://"+current_node.cord_ip+":"+current_node.cord_port+"/broadcast_replica_deletion", \
			params = {'keys':json.dumps(mark_to_remove), 'id':current_node.get_ID()})


	### Completed File Transfer
	return jsonify(
			status="Success"
	)

@app.route('/node_departure', methods=['GET', 'POST'])
def node_departure():
	if current_node.is_coord():
		id_node = request.args.get('id')
		id_node = int(id_node)
		ip_node = request.args.get('ip')
		port_node = request.args.get('port')

		### Find previous node
		previous_node_id_index = ring_ids.index(id_node) - 1
		if(previous_node_id_index == -1):
			previous_node_id_index = len(ring_ids)-1

		previous_node_id = ring_ids[previous_node_id_index]
		previous_node_ip = ring_nodes[previous_node_id]["ip"]
		previous_node_port = ring_nodes[previous_node_id]["port"]
		
		### Find next node
		next_node_id_index = ring_ids.index(id_node) + 1
		if(next_node_id_index == len(ring_ids)):
			next_node_id_index = 0

		next_node_id = ring_ids[next_node_id_index]
		next_node_ip = ring_nodes[next_node_id]["ip"]
		next_node_port = ring_nodes[next_node_id]["port"]

		### Update next_node of previous_node	
		response = requests.get(url = "http://"+str(previous_node_ip)+":"+str(previous_node_port)+"/update_next_node", params = {'id':next_node_id, 'ip':next_node_ip, 'port':next_node_port})

		### Update previous node of next_node
		response = requests.get(url = "http://"+next_node_ip+":"+next_node_port+"/update_previous_node", params = {'id':previous_node_id, 'ip':previous_node_ip, 'port':previous_node_port})
		return jsonify(
        status="Success",
		message="Bye Bye"
		)
		

	else:
		return jsonify(
        answer="Fail",
		)

@app.route('/current_depart', methods=['GET'])
def current_depart():
	app_thread = threading.Thread(target=depart_fun, args=())
	app_thread.start()
	return jsonify(
        status="Success",
		message="Bye Bye"
        #id=id_node
		)

def kill_all_processes(pid, including_parent=True):    
    parent = psutil.Process(pid)
    for child in parent.get_children(recursive=True):
        child.kill()
    if including_parent:
        # This will destroy everything
        parent.kill()

def terminate():
    me = os.getpid()
    kill_all_processes(me)

def depart_fun():
	time.sleep(1)
	### Initiate Departure Procedure

	#Inform coordinator for my departure
	if(current_node.is_coord()):
		terminate()
	
	current_node.announce_departure()
	#Before departing, send keys to the next node
	sendKeysToNextNode()

	### Notify coordinator of departure,and begin process of replica redistribution
	cord_ip = current_node.get_coord_IP()
	cord_port = current_node.get_coord_port()
	response = requests.get(url = "http://"+str(cord_ip)+":"+str(cord_port)+"/exit_confirmation", params = {'id':current_node.get_ID()})
	os.kill(int(os.getpid()), signal.SIGINT)  	

def sendKeysToNextNode():
	response = requests.post(url = "http://"+str(current_node.get_next_node()['ip'])+":"+str(current_node.get_next_node()['port'])+"/accept_files", params = {'files':json.dumps(current_node.get_files())})
	json_data = json.loads(response.text)

def signal_handler(sig, frame):
	print("You pressed Ctrl+C, the program will now exit.")
	#Announce Departure to Coordinator
	if(current_node.is_coord()):
		sys.exit(0)
	current_node.announce_departure()
	#Before departing, send keys to next node.
	sendKeysToNextNode()
	### Notify coordinator of departure,and begin process of replica redistribution
	cord_ip = current_node.get_coord_IP()
	cord_port = current_node.get_coord_port()
	response = requests.get(url = "http://"+str(cord_ip)+":"+str(cord_port)+"/exit_confirmation", params = {'id':current_node.get_ID()})
	sys.exit(0)

def thread_function():
	"""
	This thread notifies the coordinator 
	that this node will enter the ring.
	"""
	if not currentNode.is_coord():
		cord_ip = current_node.get_coord_IP()
		cord_port = current_node.get_coord_port()
		time.sleep(1)
		response = requests.get(url = "http://"+str(cord_ip)+":"+str(cord_port)+"/accept_new_node", params = {'id':current_node.get_ID(), 'ip':current_node.get_IP(), 'port':current_node.get_port()})


"""
FUNCTIONS THAT HAVE TO DO 
WITH REPLICATION
WE'LL ONLY APPLY CHAIN REPLICATION AS IT'S EASIER
AND EVENTUAL CONSISTENCY.
"""

### Chain Replication
@app.route('/insert_propagation', methods=['GET', 'POST'])
def insert_propagation():
	"""
	This is a function 
	that makes
	chain replication.
	"""
	unhashed_key = request.args.get('key')
	value = request.args.get('value')
	client_ip = request.args.get('client_ip')
	client_port = request.args.get('client_port')
	origin_id = request.args.get('origin_id')

	### Hash function ###
	m = hashlib.sha1()
	m.update(unhashed_key.encode('utf-8'))
	key = int(m.hexdigest(),16)

	#Check if this node is reponsible for replicas of origin_id
	if(not current_node.node_in_RM(origin_id)):
		return jsonify(
			status="Failed",
			message="Replica not found"
		)
	else:
		current_node.add_file_to_RM(origin_id, key, unhashed_key, value)
		response = requests.get(url = "http://"+str(current_node.get_next_node()['ip'])+":"+str(current_node.get_next_node()['port'])+"/insert_propagation", \
			params = {'key':unhashed_key, 'value':value, 'origin_id':origin_id, 'client_ip':client_ip, 'client_port':client_port})
		if(not str(response.status_code) == '200'):
			response = requests.post(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'message':"Error: file could not be saved properly"})
		else:
			json_data = json.loads(response.text)
			if(json_data['message'] == "Replica not found"):
				message = 'File saved successfully'
				response = requests.post(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'message':message})
			elif(json_data['status'] != 'Success'):
				response = requests.post(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'message':"Error: file can't be saved"})
	return jsonify(
		status="Success",
		message="Replica found"
	)

### Propagating eventual insert of file for Eventual consistency
@app.route('/insert_propagation_eventual_consistency', methods=['GET', 'POST'])
def insert_propagation_eventual_consistency():
	unhashed_key = request.args.get('key')
	value = request.args.get('value')
	origin_id = request.args.get('origin_id')

	### Hash Function ###
	m = hashlib.sha1()
	m.update(unhashed_key.encode('utf-8'))
	key = int(m.hexdigest(),16)

	#Check if this node is reponsible for replicas of origin_id
	if(not current_node.node_in_RM(origin_id)):
		return jsonify(
			status="Failed",
			message="Replica not found"
		)
	else:
		current_node.add_file_to_RM(origin_id, key, unhashed_key, value)
		response = requests.get(url = "http://"+str(current_node.get_next_node()['ip'])+":"+str(current_node.get_next_node()['port'])+"/insert_propagation_eventual_consistency", \
			params = {'key':unhashed_key, 'value':value, 'origin_id':origin_id})
	return jsonify(
		status="Success",
		message="Replica found"
	)


### Propagating file deletion for Chain Replication
@app.route('/delete_propagation', methods=['GET', 'POST'])
def delete_propagation():
	unhashed_key = request.args.get('key')
	client_ip = request.args.get('client_ip')
	client_port = request.args.get('client_port')
	origin_id = request.args.get('origin_id')

	### Produce hash of unhashed key
	m = hashlib.sha1()
	m.update(unhashed_key.encode('utf-8'))
	key = int(m.hexdigest(),16)
	key = str(key)

	#Check if this node is reponsible for replicas of origin_id
	if(not current_node.node_in_RM(origin_id)):
		return jsonify(
			status="Failed",
			message="Replica not found"
		)
	else:
		current_node.delete_file_from_RM(origin_id, key)
		response = requests.get(url = "http://"+str(current_node.get_next_node()['ip'])+":"+str(current_node.get_next_node()['port'])+"/delete_propagation", \
			params = {'key':unhashed_key, 'origin_id':origin_id, 'client_ip':client_ip, 'client_port':client_port})
		if(not str(response.status_code) == '200'):
			response = requests.post(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'message':"Error: file could not be deleted properly"})
		else:
			json_data = json.loads(response.text)
			if(json_data['message'] == "Replica not found"):
				message = 'File deleted successfully'
				response = requests.post(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'message':message})
			elif(json_data['status'] != 'Success'):
				response = requests.post(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'message':"Error: file could not be deleted properly"})
	return jsonify(
		status="Success",
		message="Replica found"
	)

### Propagating file deletion for Eventual Consistency
@app.route('/delete_propagation_eventual_consistency', methods=['GET', 'POST'])
def delete_propagation_eventual_consistency():
	unhashed_key = request.args.get('key')
	origin_id = request.args.get('origin_id')

	### Hash function ### 
	m = hashlib.sha1()
	m.update(unhashed_key.encode('utf-8'))
	key = int(m.hexdigest(),16)
	key = str(key)

	#Check if this node is reponsible for replicas of origin_id
	if(not current_node.node_in_RM(origin_id)):
		return jsonify(
			status="Failed",
			message="Replica not found"
		)
	else:
		current_node.delete_file_from_RM(origin_id, key)
	return jsonify(
		status="Success",
		message="Replica found"
	)



### Propagating query for Chain Replication
@app.route('/query_propagation', methods=['GET', 'POST'])
def query_propagation():
	unhashed_key = request.args.get('key')
	client_ip = request.args.get('client_ip')
	client_port = request.args.get('client_port')
	origin_id = request.args.get('origin_id')

	### Produce hash of unhashed key
	m = hashlib.sha1()
	m.update(unhashed_key.encode('utf-8'))
	key = int(m.hexdigest(),16)
	key = str(key)

	#Check if this node is reponsible for replicas of origin_id
	if(not current_node.node_in_RM(origin_id)):
		return jsonify(
			status="Failed",
			message="Replica not found"
		)
	else:
		response = requests.get(url = "http://"+str(current_node.get_next_node()['ip'])+":"+str(current_node.get_next_node()['port'])+"/query_propagation", \
			params = {'key':unhashed_key, 'origin_id':origin_id, 'client_ip':client_ip, 'client_port':client_port})
		if(not str(response.status_code) == '200'):
			response = requests.post(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'message':"Error: file could not be deleted properly"})
		else:
			json_data = json.loads(response.text)
			if(json_data['message'] == "Replica not found"):

				### I am the last replica manager for original node ---> return the key-value pair
				file_value = current_node.get_file_from_RM(origin_id, key)
				if not file_value is None:
					message = 'File found succesfully!'
					r_value = file_value
				else:
					message = 'File not found!'
					r_value = 'None'
				response = requests.get(url = "http://"+str(client_ip)+":"+str(client_port)+"/", \
					params = {'key':unhashed_key, 'value':r_value, 'message':message})

			elif(json_data['status'] != 'Success'):
				response = requests.post(url = "http://"+str(client_ip)+":"+str(client_port)+"/", params = {'message':"Error: file could not be deleted properly"})
	return jsonify(
		status="Success",
		message="Replica found"
	)




"""
When a node enters the ring, another node loses files from the insertion of the new node,
notifies the replica manager that he lost files.
"""
@app.route('/broadcast_replica_deletion', methods=['GET', 'POST'])
def broadcast_replica_deletion():
	if current_node.is_coord():
		id_node = request.args.get('id')
		id_node = int(id_node)
		#list with keys to be deleted
		files = request.args.get('keys')
		if(not (files is None)):
			keys_to_delete = json.loads(files)
		else:
			return jsonify(
			message = "Empty key list",
			)
		

		#Find next_node
		current_index = ring_ids.index(id_node)

		for k in range (1, current_node.replication_factor):
			index_next_node = current_index + 1
			if(index_next_node == len(ring_ids)):
				index_next_node = 0

			ip_suc = ring_nodes[ring_ids[index_next_node]]['ip']
			port_suc = ring_nodes[ring_ids[index_next_node]]['port']
			response = requests.post(url = "http://"+ip_suc+":"+port_suc+"/deletion_due_to_insertion", \
				params = {'keys':json.dumps(keys_to_delete), 'id':id_node})

		return jsonify(
			message = "Success",
		)

	else:
		return jsonify(
			message = "I am not the coordinator",
		)

"""When a new node enters,	my previous node informs me that he has given keys to his (new) previous node
 And I must delete them from his replica (that I'm keeping)
 The coordinator handles this,
 meaning the communication goes like:
 Node -> Coordinator -> RM
 """
@app.route('/deletion_due_to_insertion', methods=['GET', 'POST'])
def deletion_due_to_insertion():
	id = request.args.get('id')
	id = int(id)
	files = request.args.get('keys')

	if(not (files is None)):
		keys_to_delete = json.loads(files)
	else:
		return jsonify(
		message = "Empty key list",
		)

	for key in keys_to_delete:
		current_node.delete_file_from_RM(id,key)

	return jsonify(
		message = "Success",
		)
	

"""
 Coordinator says I am ready to send my replicas to next_nodes
 so we know it is time to seek replicas from other nodes"""
@app.route('/confirmation_to_send_replicas', methods=['GET', 'POST'])
def confirmation_to_send_replicas():
	### First take copies of replicas from next_nodes
	current_node.get_replica_sources()

	### Send replicas of my keys to next_node
	current_node.get_replicas_keeper()
	return jsonify(
	    	answer="Success"
	    	#id=id_node
		)

"""A node has departed and his keys have been forwaded to his next node
The coordinator now notifies nodes that must get a replica as a result from the departure.
If replication factor is k:
k - 1 nodes must learn a replica
node k-i (succesor) must learn the replica of node -i (previous_node)"""
@app.route('/exit_confirmation', methods=['GET', 'POST'])
def exit_confirmation():

	id_node = request.args.get('id')
	id_node = int(id_node)
	current_index = ring_ids.index(id_node)

	#Find first previous_node
	current_pred = current_index

	for k in range(1, current_node.replication_factor):
		#Find previous_node
		current_pred = current_index-1
		if(current_pred == -1):
			current_pred = len(ring_ids)-1

		#Find k-1 next_node of current index
		index_next_node = current_index + current_node.replication_factor - k
		if(index_next_node >= len(ring_ids)):
			index_next_node -= len(ring_ids) 

		### send request to current suc to learn replica from current_pred
		ip_suc = ring_nodes[ring_ids[index_next_node]]['ip']
		port_suc = ring_nodes[ring_ids[index_next_node]]['port']

		ip_pred = ring_nodes[ring_ids[current_pred]]['ip']
		port_pred = ring_nodes[ring_ids[current_pred]]['port']

		response = requests.get(url = "http://"+ip_suc+":"+port_suc+"/get_previous_node_replicas", params = {'id': ring_ids[current_pred],'ip':ip_pred, 'port':port_pred})
	
	### Also, the next_node of the departed node gets new keys. The coordinator must tell his next_nodes that have his replicas to renew them
	index_next_node_new_keys = current_index + 1
	if(index_next_node_new_keys >= len(ring_ids)):
		index_next_node_new_keys -= len(ring_ids)
	ip_suc_new_keys = ring_nodes[ring_ids[index_next_node_new_keys]]['ip']
	port_suc_new_keys = ring_nodes[ring_ids[index_next_node_new_keys]]['port']
	
	cur_suc = index_next_node_new_keys
	for k in range(1,current_node.replication_factor):
		cur_suc = cur_suc + 1
		if(cur_suc == len(ring_ids)):
			cur_suc = 0
		
		ip_suc = ring_nodes[ring_ids[cur_suc]]['ip']
		port_suc = ring_nodes[ring_ids[cur_suc]]['port']
		response = requests.get(url = "http://"+ip_suc+":"+port_suc+"/get_previous_node_replicas", \
			params = {'id': ring_ids[index_next_node_new_keys],'ip':ip_suc_new_keys, 'port':port_suc_new_keys})

	
	### Notify next_nodes of the departing node to delete his replicas
	suc_index = current_index
	for k in range(1, current_node.replication_factor):
		suc_index = suc_index + 1
		if(suc_index == len(ring_ids)):
			suc_index = 0
		ip_suc = ring_nodes[ring_ids[suc_index]]['ip']
		port_suc = ring_nodes[ring_ids[suc_index]]['port']

		response = requests.get(url = "http://"+ip_suc+":"+port_suc+"/replicas_deletion", params = {'id': ring_ids[current_index]})


	### Remove the node that departs from ring
	del ring_nodes[id_node]
	del ring_ids[ring_ids.index(id_node)]

	return jsonify(
        status="Success"
    )

### After a node has departed, coordinator informs me of a replica I have to keep
@app.route('/get_previous_node_replicas', methods=['GET', 'POST'])
def get_previous_node_replicas():
	id_of_pred = request.args.get('id')
	id_of_pred = int(id_of_pred)
	ip_node = request.args.get('ip')
	port_node = request.args.get('port')
	response = requests.get(url = "http://"+ip_node+":"+port_node+"/node_files", params = {})
	json_data = json.loads(response.text)
	replica_dict = json_data['files']
	#print(replica_dict)
	current_node.replica_manager[int(id_of_pred)] = replica_dict

	return jsonify(
        status="Success"
    )


### Coordinator hits this endpoint to inform a node about a predecessor that the node no longer needs to keep a replica
@app.route('/replicas_deletion', methods=['GET', 'POST'])
def replicas_deletion():
	id = request.args.get('id')
	id = int(id)
	current_node.delete_node_from_RM(id)
	
	return jsonify(
			status="Success"
		)


"""
This function replies with the nodes that 
the client needs to keep replicas from
"""
@app.route('/replica_nodes', methods=['GET', 'POST'])
def replica_nodes():
	if current_node.is_coord():
		id_node = request.args.get('id')
		id_node = int(id_node)
		ip_node = request.args.get('ip')
		port_node = request.args.get('port')
		
		current_index = ring_ids.index(id_node)
		pending_replicas = int(current_node.get_replication_factor()) -1
		replica_sources_list = []

		index_previous_node = current_index-1
		while(pending_replicas >0):
			if(index_previous_node == -1):
				index_previous_node = len(ring_ids)-1
			if(index_previous_node == current_index): break
			#add index_previous_node info to return node list
			replica_sources_list.append([str(ring_ids[index_previous_node]), str(ring_nodes[ring_ids[index_previous_node]]['ip']), str(ring_nodes[ring_ids[index_previous_node]]['port'])])
			#check if there are not enough nodes for the replication factor
			index_previous_node -= 1
			pending_replicas -= 1
			
		return jsonify(
        status="Success",
        replica_sources = replica_sources_list
    	)
	else:
		return jsonify(
        answer="Fail: I am not the coordinator",
        #id=id_node
    	)



"""This function replies with the nodes
 responsible for the replicas of the asking node
"""
@app.route('/replica_keepers', methods=['GET', 'POST'])
def replica_keepers():
	if current_node.is_coord():
		id_node = request.args.get('id')
		id_node = int(id_node)
		ip_node = request.args.get('ip')
		port_node = request.args.get('port')
		
		current_index = ring_ids.index(id_node)
		pending_replicas = int(current_node.get_replication_factor()) -1
		replica_keepers_list = []

		
		t_node = current_index+1
		while(pending_replicas >0):
			if(index_next_node == len(ring_ids)):
				index_next_node = 0
			if(index_next_node == current_index): break
			#add index_previous_node info to return node list
			replica_keepers_list.append([str(ring_ids[index_next_node]), str(ring_nodes[ring_ids[index_next_node]]['ip']), str(ring_nodes[ring_ids[index_next_node]]['port'])])
			#check if there are not enough nodes for the replication factor
			index_next_node += 1
			pending_replicas -= 1
			
		return jsonify(
        status="Success",
        replica_keepers = replica_keepers_list
    	)
	else:
		return jsonify(
        answer="Fail: I am not the coordinator",
        #id=id_node
    )




"""
MAIN FUNCTION
"""


"""
INPUT ARGUMENTS FOR COORDINATOR: 
port & replication factor & consistency_type (linearizability or eventual consistency)

"""

"""
INPUT ARGUMENTS FOR SLAVE: 
port & replication factor &  consistency_type & coordinator_ip & coordinator_port 

"""
if __name__ == '__main__':
	#change domain to local ip (e.g. 192.168.0.2) if running on Okeanos.
	domain = 'localhost'
	port = sys.argv[1]
	if len(sys.argv) == 6: #Normal node if it has 6 arguments
		currentNode = dn.Node(domain,port, cord_ip = sys.argv[4], cord_port = sys.argv[5], replication_factor = sys.argv[2], consistency_type = sys.argv[3])
	else: #else it's the coordinator
		currentNode = dn.Node(domain,port, replication_factor = sys.argv[2], consistency_type = sys.argv[3])
	if currentNode.is_coord():
		ring_nodes[currentNode.get_ID()] = {"ip" : currentNode.get_IP(), "port" : currentNode.get_port()}
		bisect.insort(ring_ids, currentNode.get_ID())
	current_node = currentNode

	#Creat a new thread to request coordinator for files
	app_thread = threading.Thread(target=thread_function, args=())
	app_thread.start()
	### exit signal
	signal.signal(signal.SIGINT, signal_handler)
	#change host to local ip (e.g. 192.168.0.2) if running on Okeanos.
	app.run(host='localhost', threaded = True, port=int(sys.argv[1]))




	




