import hashlib,urllib,time,signal,sys,grequests,requests,sys,bisect,json
from flask import Flask, make_response, jsonify, Response
from flask import request
from gevent import monkey
monkey.patch_all()
from random import randint
app = Flask(__name__)


COORD_IP = ''
COORD_PORT = ''
MY_IP = ''
MENU_MODULE_PORT = ''
BASE_PORT = 5000
CHORD_NODES = {}
NODE_IP = {
    4 : '192.168.0.2',
    3 : '192.168.0.3',
    2 : '192.168.0.4',
    1 : '192.168.0.5',
    0 : '192.168.0.6'
}

query_reply_list = []
insert_reply_list = []


@app.route('/', methods=['GET'])
def listen_simple_query():
    key = request.args.get('key')
    value = request.args.get('value')
    print("Received reply: key={}, value={}".format(key,value))
    return jsonify(
        status="ok"
    )


# A simple task to do to each response object
def do_something(response):
    print(response.url)



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@app.route('/insert_from_file', methods=['GET'])
def insert_from_file():
    file = request.args.get('file')

    # Using readlines()
    file1 = open(file, 'r')
    Lines = file1.readlines()
    
    async_list = []
    # Strips the newline character
    for line in Lines:
        line = line.strip('\n')
        list = line.split(', ')
        key = list[0]
        value = list[1]
        action_item = grequests.get("http://"+str(COORD_IP)+":"+str(COORD_PORT)+"/insert", hooks = {'response' : do_something}, \
             params = {'key':key, 'value':value, 'client_ip': MY_IP, 'client_port':sys.argv[1]+"/get_insert_result"})

        # async task
        async_list.append(action_item)
    # Do our list of things to do via async
    chunk_list = chunks(async_list, 100)
    for as_list in chunk_list:
        grequests.map(as_list, stream=False)
    return jsonify(
        status="ok"
    )

@app.route('/run_requests_from_file', methods=['GET'])
def run_requests_from_file():
    file = request.args.get('file')

    # Using readlines()
    file1 = open(file, 'r')
    Lines = file1.readlines()
    for line in Lines:
        line = line.strip('\n')
        list = line.split(', ')

        type = list[0]
        key = list[1]
        value = None
        if type == 'insert': value = list[2]
        if type == 'query':
            response = requests.get(url = "http://"+'localhost'+":"+'5000'+"/query", params = {'key':key, 'client_ip':'127.0.0.1', 'client_port':sys.argv[1]})
            return response.text
        
        else:
            response = requests.get(url = "http://"+'localhost'+":"+'5000'+"/insert", \
                params = {'key':key, 'value':value, 'client_ip': MY_IP, 'client_port':sys.argv[1]+"/get_insert_result"})

    return jsonify(
        status="ok"
    )

@app.route('/get_insert_result/', methods=['GET','POST'])
def get_insert_result():
    message = request.args.get('message')

    print("Received reply: message={}".format(message))
    return jsonify(
        status="ok"
    )

@app.route('/', methods=['POST'])
def listen_star_query():
    node_id = request.args.get('node_id')
    files = request.args.get('files')
    json_file = json.loads(files)
    print(" Query * reply: Node={}".format(node_id))
    for key, value in json_file.items():
        print("File:{}  --- Value:{}".format(value[0],value[1]))
    return jsonify(
        status="ok"
    )

@app.route('/query', methods=['GET', 'POST'])
def query():
    unhashed_key = request.args.get('key')
    ip = request.args.get('ip') 
    port = request.args.get('port')
    response = requests.get(url = "http://"+str(ip)+":"+str(port)+"/query", params = {'key':unhashed_key, 'client_ip':'127.0.0.1', 'client_port':sys.argv[1]})
    return response.text

@app.route('/delete', methods=['GET', 'POST'])
def delete():
    unhashed_key = request.args.get('key')
    ip = request.args.get('ip') 
    port = request.args.get('port')
    response = requests.get(url = "http://"+str(ip)+":"+str(port)+"/delete", \
        params = {'key':unhashed_key, 'client_ip':'127.0.0.1', 'client_port':sys.argv[1]+"/get_delete_result"})
    return response.text

@app.route('/get_delete_result/', methods=['GET'])
def get_delete_result():
    message = request.args.get('message')
    print("Received reply: message={}".format(message))
    return jsonify(
        status="ok"
    )

"""
CLI END-POINTS
"""

#INSERT ENDPOINT
@app.route('/insert_cli', methods=['GET'])
def insert_cli():
    ip = request.args.get('ip')
    port = request.args.get('port')
    key = request.args.get('key')
    value = request.args.get('value')

    if not ip or not port:  #Choose a random node
        random_node = randint(0, len(CHORD_NODES.keys())-1)
        ip = CHORD_NODES[random_node]['ip']
        port = CHORD_NODES[random_node]['port']

    #Pass query to the corresponding node
    response = requests.get(url = "http://"+str(ip)+":"+str(port)+"/insert", \
        params = {'key':key, 'value':value, 'client_ip':MY_IP, 'client_port':sys.argv[1]+"/get_result_cli"})

    return jsonify(
        status="Success"
    )

@app.route('/insert_from_file_cli', methods=['GET'])
def insert_from_file_cli():
    file = request.args.get('file')

    # Using readlines()
    file1 = open(file, 'r')
    Lines = file1.readlines()
    
    async_list = []
    # Strips the newline character
    for line in Lines:
        line = line.strip('\n')
        list = line.split(', ')

        key = list[0]
        value = list[1]
        #Choose random node
        random_node = randint(0, len(CHORD_NODES.keys())-1)
        ip = CHORD_NODES[random_node]['ip']
        port = CHORD_NODES[random_node]['port']
        action_item = grequests.get("http://"+str(ip)+":"+str(port)+"/insert", hooks = {'response' : do_something}, \
             params = {'key':key, 'value':value, 'client_ip': MY_IP, 'client_port':sys.argv[1]+"/get_insert_result"})

        # async task
        async_list.append(action_item)
    chunk_list = chunks(async_list, 100)
    for as_list in chunk_list:
        grequests.map(as_list, stream=False)
        #time.sleep(0.2)
    #Pass message to menu module
    response = requests.get(url = "http://"+str(MY_IP)+":"+str(MENU_MODULE_PORT)+"/", \
        params = {'message':"All files uploaded to the DHT successfully"})

    return jsonify(
        status="ok"
    )


#DELETE ENDPOINT
@app.route('/delete_cli', methods=['GET'])
def delete_cli():
    ip = request.args.get('ip')
    port = request.args.get('port')
    key = request.args.get('key')

    # ### DELETE NODE (if present) from CHORD_NODES
    # for i in CHORD_NODES.keys():
    #     if CHORD_NODES[i]['ip'] == ip and CHORD_NODES[i]['port'] == port:
    #         del CHORD_NODES[i]
    #         break

    if not ip or not port:
        random_node = randint(0, len(CHORD_NODES.keys())-1)
        ip = CHORD_NODES[random_node]['ip']
        port = CHORD_NODES[random_node]['port']


    #Pass query to the corresponding node
    response = requests.get(url = "http://"+str(ip)+":"+str(port)+"/delete", \
        params = {'key':key, 'client_ip':MY_IP, 'client_port':sys.argv[1]+"/get_result_cli"})

    return jsonify(
        status="Success"
    )

#QUERY ENDPOINT
@app.route('/query_cli', methods=['GET'])
def query_cli():
    ip = request.args.get('ip')
    port = request.args.get('port')
    key = request.args.get('key')

    if not ip or not port:
        random_node = randint(0, len(CHORD_NODES.keys())-1)
        ip = CHORD_NODES[random_node]['ip']
        port = CHORD_NODES[random_node]['port']


        #Pass query to the corresponding node
        response = requests.get(url = "http://"+str(ip)+":"+str(port)+"/query", \
            params = {'key':key, 'client_ip':MY_IP, 'client_port':sys.argv[1]+"/get_query_result_cli"})

    return jsonify(
        status="Success"
    )

#GENERIC RESULT ENDPOINT
@app.route('/get_result_cli/', methods=['GET', 'POST'])
def get_result_cli():
    message_s = request.args.get('message')
    print("Received reply: message={}".format(message_s))

    #Pass message to menu module
    response = requests.get(url = "http://"+str(MY_IP)+":"+str(MENU_MODULE_PORT)+"/", \
        params = {'message':message_s})


    return jsonify(
        status='Success',
        message = message_s
    )

#QUERY RESULT ENDPOINT
@app.route('/get_query_result_cli/', methods=['GET', 'POST'])
def get_query_result_cli():
    key = request.args.get('key')
    value = request.args.get('value')
    if(not key and not value):
        files = request.args.get('files')
        response = requests.post(url = "http://"+str(MY_IP)+":"+str(MENU_MODULE_PORT)+"/", \
        params = {'files':files})
    else:
        print("Received reply: key={} - value={}".format(key, value))

        #Pass message to menu module
        response = requests.get(url = "http://"+str(MY_IP)+":"+str(MENU_MODULE_PORT)+"/", \
            params = {'key':key, 'value':value})


    return jsonify(
        status='Success',
    )

#QUERY ENDPOINT
@app.route('/view_files_cli', methods=['GET'])
def view_files_cli():
    ip = request.args.get('ip')
    port = request.args.get('port')


    #Pass query to the corresponding node
    response = requests.get(url = "http://"+str(ip)+":"+str(port)+"/node_files", \
        params = {})

    #Pass message to menu module
    response = requests.post(url = "http://"+str(MY_IP)+":"+str(MENU_MODULE_PORT)+"/", \
        params = {'message' : response.text})

    return jsonify(
        status="Success"
    )

#QUERY ENDPOINT
@app.route('/departure_cli', methods=['GET'])
def departure_cli():
    ip = request.args.get('ip')
    port = request.args.get('port')
    ## Remove departing node from CHORD_NODES
    ### DELETE NODE (if present) from CHORD_NODES
    for i in CHORD_NODES.keys():
        if CHORD_NODES[i]['ip'] == ip and CHORD_NODES[i]['port'] == port:
            del CHORD_NODES[i]
            break

    #Pass query to the corresponding node
    response = requests.get(url = "http://"+str(ip)+":"+str(port)+"/current_depart", \
        params = {})
    
    #Pass message to menu module
    response = requests.get(url = "http://"+str(MY_IP)+":"+str(MENU_MODULE_PORT)+"/", \
        params = {'message':response.text})

    return jsonify(
        status="Success",
        node_response = response.text
    )

#RUN BATCH REQUESTS FROM FILE
@app.route('/run_requests_from_file_cli', methods=['GET'])
def run_requests_from_file_cli():
    file = request.args.get('file')

    # Using readlines()
    file1 = open(file, 'r')
    Lines = file1.readlines()
    query_reply_list = []
    insert_reply_list = []
    for line in Lines:
        line = line.strip('\n')
        list = line.split(', ')
        type = list[0]
        key = list[1]
        value = None
        if type == 'insert': value = list[2]

        #Choose a random node
        random_node = randint(0, len(CHORD_NODES.keys())-1)
        ip = CHORD_NODES[random_node]['ip']
        port = CHORD_NODES[random_node]['port']

        #Pass query to the corresponding node
        if type == 'query':
            response = requests.get(url = "http://"+str(ip)+":"+str(port)+"/query", params = {'key':key, 'client_ip':MY_IP, 'client_port':sys.argv[1]+"/get_batch_query_result_cli"})
        
        else:
            response = requests.get(url = "http://"+str(ip)+":"+str(port)+"/insert", \
                params = {'key':key, 'value':value, 'client_ip': MY_IP, 'client_port':sys.argv[1]+"/get_insert_result_cli"})

    #Pass message to menu module
    response = requests.get(url = "http://"+str(MY_IP)+":"+str(MENU_MODULE_PORT)+"/", \
        params = {'query_reply_list':query_reply_list, 'insert_reply_list': insert_reply_list})

    return jsonify(
        status="ok"
    )

@app.route('/get_batch_query_result_cli/', methods=['GET','POST'])
def get_batch_query_result_cli():
    key = request.args.get('key')
    value = request.args.get('value')

    if(not key and not value):
        files = request.args.get('files')
        query_reply_list.append(str(files))
    else:

        print("Received reply: key={} - value={}".format(key,value))

        #Pass message to menu module
        response = requests.get(url = "http://"+str(MY_IP)+":"+str(MENU_MODULE_PORT)+"/", \
            params = {'message':"Received reply: key={} - value={}".format(key,value)})

        # query_reply_list.append("Received reply: key={} - value={}".format(key,value))
    return jsonify(
        status="ok"
    )


@app.route('/get_insert_result_cli/', methods=['GET','POST'])
def get_insert_result_cli():
    message = request.args.get('message')

    print("Received reply: message={}".format(message))

    #Pass message to menu module
    response = requests.get(url = "http://"+str(MY_IP)+":"+str(MENU_MODULE_PORT)+"/", \
        params = {'message':message})

    # insert_reply_list.append(message)
    return jsonify(
        status="ok"
    )

#RUN BATCH QUERIES FROM FILE
@app.route('/run_queries_from_file_cli', methods=['GET'])
def run_queries_from_file_cli():
    file = request.args.get('file')

    # Using readlines()
    file1 = open(file, 'r')
    Lines = file1.readlines()
    query_reply_list = []
    # Strips the newline character
    for line in Lines:
        line = line.strip('\n')
        
        key = line

        #Choose a random node
        random_node = randint(0, len(CHORD_NODES.keys())-1)
        ip = CHORD_NODES[random_node]['ip']
        port = CHORD_NODES[random_node]['port']
        
        response = requests.get(url = "http://"+str(ip)+":"+str(port)+"/query", params = {'key':key, 'client_ip':MY_IP, 'client_port':sys.argv[1]+"/get_batch_query_result_cli"})

    #Pass message to menu module
    response = requests.get(url = "http://"+str(MY_IP)+":"+str(MENU_MODULE_PORT)+"/", \
        params = {'query_reply_list':query_reply_list})

    return jsonify(
        status="ok"
    )


#TOPOLOGY ENDPOINT
@app.route('/topology_cli', methods=['GET'])
def print_topology():
    #print(CHORD_NODES)
    ip = request.args.get('ip')
    port = request.args.get('port')

    if not ip or not port:
        ip = COORD_IP
        port = COORD_PORT

    #Pass message to menu module
    response = requests.get(url = "http://"+str(ip)+":"+str(port)+"/topology", \
        params = {})

    #Pass message to menu module
    response = requests.get(url = "http://"+str(MY_IP)+":"+str(MENU_MODULE_PORT)+"/extract_overlay", \
        params = {'message':response.text, 'coordinator_ip':COORD_IP, 'coordinator_port':COORD_PORT})

    return jsonify(
        status="ok",
        topology=CHORD_NODES
    )

""" 
Parameters: 1: port for cli, 2: number of chord nodes, 3: MY_IP, 4.Menu Module Port (the IP is shared)

"""
if __name__ == '__main__':
    MY_IP = sys.argv[3]
    MENU_MODULE_PORT = sys.argv[4]
    ### Build Node Dictionary
    number_of_nodes = int(sys.argv[2])
    k = number_of_nodes
    n = len(NODE_IP.keys())-1
    port_base = 0
    for i in range(k):
        if(n<0):
            n = len(NODE_IP.keys())-1
            port_base +=1


        CHORD_NODES[i] = {'ip':NODE_IP[n], 'port':BASE_PORT + port_base}
        if(i==0):
            COORD_IP = NODE_IP[n]
            COORD_PORT = BASE_PORT + port_base
            n -=1
    
    print(CHORD_NODES)
    app.run(host=MY_IP, threaded = True, port=int(sys.argv[1]))