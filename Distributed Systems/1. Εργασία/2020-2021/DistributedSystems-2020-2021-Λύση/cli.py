from cli_receiver import do_something, run_requests_from_file
from flask import Flask, make_response, jsonify, Response
from flask import request
from flask_script import Manager, Server
import json,threading,signal,sys,shlex,requests,bisect,logging
from datetime import datetime
import dht_node as dn
app = Flask(__name__)

###Disable annoying messages.
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

""" Receive messages the cli backend
"""
@app.route('/', methods=['GET', 'POST'])
def print_server_message_contents():    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    multi_dict = request.args
    for key in multi_dict:
        if request.method == 'POST':
            file_list = json.loads(multi_dict.get(key))
            print("Number of received files:{} --- Time".format(len(file_list.keys()), current_time))
        print(multi_dict.get(key)+ " ---- Time:"+ str(current_time))
        #print(multi_dict.getlist(key))
	
    return jsonify(
	    status="Success"
	)


@app.route('/extract_overlay', methods=['GET', 'POST'])
def extract_overlay():
    coord_ip = request.args.get('coordinator_ip')
    coord_port = request.args.get('coordinator_port')

    message = request.args.get('message')
    json_overlay = json.loads(message)

    nodes = json_overlay['nodes']
    print("The Coordinator is : {}:{}\n".format(coord_ip, coord_port))
    for item in nodes:
        print("{} --- {}".format(item, nodes[item]))
	
    return jsonify(
	    status="Success"
	)
    

def signal_handler(sig, frame):
	print('You pressed Ctrl+C!')	
	print("Bye bye!")
	sys.exit(0)

def insert(params, server_cli_ip, server_cli_port):
    # Pass insert request to Backend
    if not (len(params) == 3 or len(params) == 5):
        print("Oops! Wrong Command, please press 'help'")

    ip = None
    port = None
    if len(params) == 5:
        ip = params[3]
        port = params[4]

    key = params[1]
    value = params[2]
    #escape characters because we can't query songs with spaces and other characters.
    if key.startswith('"') and key.endswith('"'):
        key = key[1:-1]
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    #Pass message to server side cli
    response = requests.get(url = "http://"+server_cli_ip+":"+server_cli_port+"/insert_cli", \
        params = {'key':key, 'value':value, 'ip':ip, 'port':port})
    

def delete(params, server_cli_ip, server_cli_port):
    # Pass insert request to SERVER SIDE CLIE
    if not (len(params) == 2 or len(params) == 4):
        print("Oops! Wrong Command, please press 'help'")

    ip = None
    port = None
    if len(params) == 4:
        ip = params[2]
        port = params[3]

    key = params[1]
    if key.startswith('"') and key.endswith('"'):
        key = key[1:-1]
    #Pass message to server side cli
    response = requests.get(url = "http://"+server_cli_ip+":"+server_cli_port+"/delete_cli", \
        params = {'key':key, 'ip':ip, 'port':port})
    

def query(params, server_cli_ip, server_cli_port):
    # Pass insert request to SERVER SIDE CLIE
    if not (len(params) == 2 or len(params) == 4):
        print("Oops! Wrong Command, please press 'help'")

    ip = None
    port = None
    if len(params) == 4:
        ip = params[2]
        port = params[3]

    key = params[1]
    if key.startswith('"') and key.endswith('"'):
        key = key[1:-1]
    #Pass message to server side cli
    response = requests.get(url = "http://"+server_cli_ip+":"+server_cli_port+"/query_cli", \
        params = {'key':key, 'ip':ip, 'port':port})
    

def depart(params, server_cli_ip, server_cli_port):
    # Pass insert request to SERVER SIDE CLIE
    if not (len(params) == 3):
        print("Oops! Wrong Command, please press 'help'")


    ip = params[1]
    port = params[2]

    #Pass message to server side cli
    response = requests.get(url = "http://"+server_cli_ip+":"+server_cli_port+"/departure_cli", \
        params = {'ip':ip, 'port':port})

def insert_from_file(params, server_cli_ip, server_cli_port):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # Pass insert request to SERVER SIDE CLIE
    if not (len(params) == 2):
        print("Oops! Wrong Command, please press 'help'")

    print("Starting file insertion, at time: {}".format(current_time))

    file_name = params[1]
    #Pass message to server side cli
    response = requests.get(url = "http://"+server_cli_ip+":"+server_cli_port+"/insert_from_file_cli", \
        params = {'file':file_name})

def request_file(params, server_cli_ip, server_cli_port):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # Pass insert request to SERVER SIDE CLIE
    if not (len(params) == 2):
        print("Oops! Wrong Command, please press 'help'")

    print("Initiating batch requesting: {}".format(current_time))
    file_name = params[1]
    #Pass message to server side cli
    response = requests.get(url = "http://"+server_cli_ip+":"+server_cli_port+"/run_requests_from_file_cli", \
        params = {'file':file_name})

def query_file(params, server_cli_ip, server_cli_port):
    # Pass insert request to SERVER SIDE CLIE
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    if not (len(params) == 2):
        print("Oops! Wrong Command, please press 'help'")

    print("Initiating batch querying: {}".format(current_time))
    file_name = params[1]
    #Pass message to server side cli
    response = requests.get(url = "http://"+server_cli_ip+":"+server_cli_port+"/run_queries_from_file_cli", \
        params = {'file':file_name})

def view_node_files(params, server_cli_ip, server_cli_port):
    # Pass insert request to SERVER SIDE CLIE
    if not (len(params) == 3):
        print("Oops! Wrong Command, please press 'help'")

    ip = params[1]
    port = params[2]

    #Pass message to server side cli
    response = requests.get(url = "http://"+server_cli_ip+":"+server_cli_port+"/view_files_cli", \
        params = {'ip':ip, 'port':port})
    
#This will print help commands
def help(params):
    a = """ 
    insert <key> <value> <ip>(optional) <port>(optional) \n
    ---------------------------------------------------\n
    delete <key> <ip>(oprtional) <port>(optional) \n
    ---------------------------------------------------\n
    query <key> <ip>(oprtional) <port>(optional)\n
    ---------------------------------------------------\n
    depart <ip> <port>\n
    ---------------------------------------------------\n
    overlay <ip>(optional) <port>(optional)\n
    ---------------------------------------------------\n
    insert_from_file <file_name> \n
    ---------------------------------------------------\n
    view_node_files  <ip> <port> \n
    ---------------------------------------------------\n
    query_from_file <file_name>\n
    ---------------------------------------------------\n
    request_from_file <file_name>\n
    ---------------------------------------------------\n
    help\n
    ---------------------------------------------------\n
    --->ip and port for a specific node, you can see the ports and ip's for each node by pressing 'overlay'\n
    """
    print(a)



def overlay(params, server_cli_ip, server_cli_port):
    # Pass insert request to SERVER SIDE CLIE
    if not (len(params) == 1 or len(params) == 3):
        print("Oops! Wrong Command, please press 'help'")

    ip = None
    port = None

    if len(params) == 3:
        ip = params[1]
        port = params[2]

    #Pass message to server side cli
    response = requests.get(url = "http://"+server_cli_ip+":"+server_cli_port+"/topology_cli", \
        params = {'ip':ip, 'port':port})
    

def thread_menu(server_cli_ip, server_cli_port):
    # Print Welcome Message:
    b = 'Welcome to our ToyChord CLI!'
    print(b)


    while True:
        command = input("Please enter a command, press 'help' for available commands.\n")
        #params = command.split(' ')
        params = shlex.split(command)
        #ToyChord client basic commands
        if params[0] == "insert":
            insert(params, server_cli_ip, server_cli_port)
        elif params[0] == "delete":
            delete(params, server_cli_ip, server_cli_port)
        elif params[0] == "query":
            query(params, server_cli_ip, server_cli_port)
        elif params[0] == "depart":
            depart(params, server_cli_ip, server_cli_port)
        elif params[0] == "overlay":
            overlay(params, server_cli_ip, server_cli_port)
        elif params[0] == "help":
            help(params)
        #Extra Commands that are helpful.
        elif params[0] == "view_node_files":
            view_node_files(params, server_cli_ip, server_cli_port)
        elif params[0] == "insert_from_file":
            insert_from_file(params, server_cli_ip, server_cli_port)
        elif params[0] == "request_from_file":
            request_file(params, server_cli_ip, server_cli_port)
        elif params[0] == "query_from_file":
            query_file(params, server_cli_ip, server_cli_port)
        else:
            print("Oops! This command doesn't exist, please press 'help'")

"""
input parameters: 1. server_cli_ip, 2. server_cli_port, 3. my port
 usually: 192.168.0.2 - 6000 - 7000
"""
if __name__ == '__main__':
    ### MUST BE THE SAME WITH MY IP (both modules run on the same machine)
	server_cli_ip = sys.argv[1]
	server_cli_port = sys.argv[2]
	#New thread to request coordinator for files
	app_thread = threading.Thread(target=thread_menu, args=(server_cli_ip, server_cli_port))
	app_thread.start()
	### Exit Signal
	signal.signal(signal.SIGINT, signal_handler)
	app.run(host=server_cli_ip,threaded = True, port=int(sys.argv[3]))
