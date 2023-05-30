from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from collections import OrderedDict
import datetime
from hashlib import sha256
from uuid import uuid4
import pickle
import requests
from urllib.parse import urlparse

BLOCKCHAIN_URL = '127.0.0.1:8080'

class Storage:
    def __init__(self, path):
        self.base_path = path
        self.blockchain_url = BLOCKCHAIN_URL
        # add genesis block to blockchain
        

    def link_to_resource(self, rid, user_public_key, user_url='127.0.0.1:5000'):
        # in reality this should be handled differently
        link = self.base_path + str(rid)
        r = requests.post(url=BLOCKCHAIN_URL+'/access_link',data={'link':link,'rid':rid})# ENCRYPT BY USER ID
        return r.status_code
    

    def log_transaction(self, rid, operation, user_public_key):
        # in reality this should be handled differently
        link = self.base_path + str(rid)
        r = requests.post(url=BLOCKCHAIN_URL+'/storage/log',data={'link':link,'rid':rid})# ENCRYPT BY USER ID
        return r.status_code

storage = Storage('./../files/')
app = Flask(__name__)
CORS(app)


@app.route('/link_transaction', methods=["POST"])
def read_transaction():
    values = request.form
    other_params = {'hash_ct': values['hash_ct'], 'accept_reject': values['accept_reject']}
    transaction_results = storage.link_to_resource()
    log_transaction_results = storage.log_transaction()
    response = {'message': 'Transaction will be added to the Block ' + str(transaction_results)}
    return jsonify(response), 201




if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Storage always runs on this port:
    parser.add_argument('-p', '--port', default=3005, type=int, help="port to listen to")
    args = parser.parse_args()
    port = args.port
    app.run(host='127.0.0.1', port=port, debug=True)


