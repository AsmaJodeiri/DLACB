#!/usr/bin/env python
# coding: utf-8

import os
import time
import importlib

import json
from collections import OrderedDict
import logging
import argparse
import numpy as np

import random
from numpy import loadtxt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from os import path

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from collections import OrderedDict
import datetime
from hashlib import sha256
from Crypto.Hash import SHA
from uuid import uuid4
import pickle
import requests
from urllib.parse import urlparse

import os
import time
import importlib
import json
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random

from numpy import loadtxt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from os import path

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

debug = False

MINING_SENDER = 'The Blockchain'
MINING_REWARD = 1
MINING_DIFFICULTY = 2

STORAGE_URL = '127.0.0.1:3005'

def hash_func(transactions):
    data = str(transactions)
    hash_data = sha256(data.encode('utf-8')).hexdigest()
    return hash_data


class Block:
    def __init__(self, transactions, nonce, previous_hash):
        self.timestamp = datetime.datetime.now().timestamp()
        self.transactions = transactions
        self.nonce = nonce
        self.previous_hash = previous_hash
        self.hash = hash_func(transactions)\
        


class Blockchain:
    def __init__(self):
        self.chain = []
        self.transactions = []
        self.node_id = str(uuid4()).replace('-', '')
        self.nodes = set()
        self.logs = 0
        # add genesis block to blockchain
        self.genesis()

    def genesis(self):
        # create first block of chain as genesis
        genesis_block = Block('', '', '----')
        genesis_block = {
            'transactions': genesis_block.transactions,
            'timestamp': genesis_block.timestamp,
            'hash': genesis_block.hash,
            'nonce': genesis_block.nonce
        }
        self.chain.append(genesis_block)

    def register_node(self, node_url):
        parsed_url = urlparse(node_url)
        if parsed_url.netloc:
            self.nodes.add(parsed_url.netloc)
        elif parsed_url.path:
            self.nodes.add(parsed_url.path)
        else:
            raise ValueError('Invalid URL')

    def add_block(self, nonce):
        # create new block object
        block = Block(self.transactions, nonce, self.chain[-1]['hash'])
        block = {
            'transactions': block.transactions,
            'timestamp': block.timestamp,
            'hash': block.hash,
            'previous_hash': block.previous_hash,
            'nonce': nonce
        }
        # add new block to chain
        self.chain.append(block)
        # add all transactions to new block
        self.transactions = []
        return block

    def valid_proof(self, transactions, last_hash, nonce, difficulty=MINING_DIFFICULTY):
        guess = (str(transactions) + str(last_hash) + str(nonce)).encode('utf8')
        h = sha256(guess)
        guess_hash = h.hexdigest()
        return guess_hash[:difficulty] == '0' * difficulty

    def proof_of_work(self):
        last_block = self.chain[-1]
        last_hash = last_block['hash']
        nonce = 0
        while self.valid_proof(self.transactions, last_hash, nonce) is False:
            nonce += 1
        return nonce

    def valid_chain(self, chain):
        for ind in range(1, len(chain)):
            # check previous block hash with current block previous_hash field
            if chain[ind]['previous_hash'] != chain[ind-1]['hash']:
                return False
        return True

    def resolving_conflict(self):
        neighbours = self.nodes
        new_chain = None
        max_length = len(self.chain)
        for node in neighbours:
            response = requests.get('http://' + node + '/chain')
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                print(self.valid_chain(chain))
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
        if new_chain:
            self.chain = new_chain
            return True
        return False


    def verify_transaction_signature(self, public_key, signature, transaction):
        # check the signature of transaction to validate sender
        with open('../keys/public_secret_key.pickle', 'rb') as handle:
            public_and_sk = pickle.load(handle)
        try:
            key = public_and_sk[public_key]
        except:
            return False
        h = SHA.new(str(transaction).encode('utf8'))
        str_hex = str(h.hexdigest())
        h = SHA.new(key.encode('utf8'))
        str_hex1 = str(h.hexdigest())
        sign = str_hex + str_hex1
        if signature == sign:
            return True
        else:
            return False

    def replace_chain(self, new_blockchain):
        if len(new_blockchain.chain) <= len(self.chain):
            return self
        if new_blockchain.is_valid_chain():
            self.chain = new_blockchain.chain
        return self

    def submit_transaction(self, sender_public_key, signature, timestamp, other_params):
        """
        add new transaction to the blockchain.
        There are 5 different types of transactions for this work.
        1.setup transaction 2.Read transaction 3.write transaction 4. Access transaction 5. Mining transaction

        we do not have same structure for different transaction types. for this reason, we need postprocessing after
        reaching transactions to the nodes.
        """
        transaction = dict()
        if 'universal_attribute' in list(other_params.keys()):
            # setup transaction
            transaction = OrderedDict({
                'admin_public_key': sender_public_key,
                'timestamp': str(timestamp),
                'universal_attribute': other_params['universal_attribute']
            })

        elif 'reward' in list(other_params.keys()):
            # mining transaction
            transaction = OrderedDict({
                'miner_public_key': sender_public_key,
                'timestamp': str(timestamp),
                'reward': other_params['reward']
            })

        elif 'user_attribute' in list(other_params.keys()):
            # access transaction
            transaction = OrderedDict({
                "user_public_key": sender_public_key,
                'user_attribute': other_params['user_attribute']
            })

        elif 'Acc' in list(other_params.keys()):
            # write transaction
            transaction = OrderedDict({
                "sender_public_key": sender_public_key,
                'ct': other_params['ct'],
                'hash_ct': other_params['hash_ct'],
                'Acc': other_params['Acc']
            })

        else:
            # read transaction
            transaction = OrderedDict({
                "reader_public_key": sender_public_key,
                'file_hash': other_params['hash_ct'],
                'timestamp': str(timestamp),
                'accept_reject': other_params['accept_reject']
            })

        tran = transaction.copy()
        if 'sender_public_key' in transaction.keys():
            # write transaction
            tran['type'] = 'Write'
            tran['accept_reject'] = '-'
            tran['public_key'] = tran.pop('sender_public_key')

        if 'reader_public_key' in transaction.keys():
            # read transaction
            tran['type'] = 'Read'
            tran['public_key'] = tran.pop('reader_public_key')
            tran['hash_ct'] = tran.pop('file_hash')

        if 'user_public_key' in transaction.keys():
            # Access transaction
            tran['type'] = 'Access'
            tran['public_key'] = tran.pop('user_public_key')
            tran['hash_ct'] = '-'
            tran['accept_reject'] = '-'

        if 'admin_public_key' in transaction.keys():
            # setup transaction
            tran['type'] = 'Setup'
            tran['public_key'] = tran.pop('admin_public_key')
            tran['hash_ct'] = '-'
            tran['accept_reject'] = '-'

        if 'miner_public_key' in transaction.keys():
            tran['type'] = 'mine'
            tran['public_key'] = tran.pop('miner_public_key')
            tran['hash_ct'] = '-'
            tran['accept_reject'] = '-'

        if signature != '':
            signature_verification = self.verify_transaction_signature(sender_public_key, signature, transaction)
            if signature_verification:
                self.transactions.append(tran)
                link = self.storage_request(other_params['rid'],sender_public_key )
                return len(self.chain) + 1, link
            else:
                return False
        else:
            self.transactions.append(tran)

    def storage_request(rid, client_public_key, client_url='NONE'):
        data = {'rid':rid, 'pkey':client_public_key, 'url':client_url}

        
        # sending get request and saving the response as response object
        r = requests.get(url = STORAGE_URL+'/log_transaction', params = data)
        if r.status_code == 201:
            ruser = requests.post()
        else:
            pass
        return r
        

    def log(self, user_public_key, rid, operation):
        with open(str(self.logs+1),'w') as f:
            f.write(user_public_key+'|'+str(rid)+'|'+str(operation))    

        return    


blockchain = Blockchain()
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/configure')
def configure():
    return render_template('configure.html')


@app.route('/transactions/get', methods=['GET'])
def get_transaction():
    transactions = blockchain.transactions
    response = {'transactions': transactions}
    return jsonify(response), 200


@app.route('/chain', methods=['GET'])
def get_chain():
    response = {
        'chain' : blockchain.chain,
        'length': len(blockchain.chain)
    }
    return jsonify(response), 200


@app.route('/mine', methods=['GET'])
def mine():
    nonce = blockchain.proof_of_work()
    # add mining info to blockchain
    other_params = {'reward': MINING_REWARD}
    if blockchain.transactions != []:
        blockchain.submit_transaction(MINING_SENDER, '', '', other_params)
        # create and add block to blockchain
        block = blockchain.add_block(nonce)
    else:
        response = {'message': 'Invalid transaction/signature'}
        return jsonify(response), 406

    response = {
        'message': 'new block created',
        'transactions': block['transactions'],
        'nonce': block['nonce'],
        'timestamp': block['timestamp']
    }
    return jsonify(response), 200

@app.route('/setup_transaction', methods=["POST"])
def system_setup_transaction():
    values = request.form
    other_params = {'universal_attribute': values['confirmation_universal_attribute']}

    transaction_results = blockchain.submit_transaction(values['confirmation_admin_public_key'],
                                                        values['transaction_signature'],
                                                        values['confirmation_timestamp'],
                                                        other_params)
    if transaction_results == False:
        response = {'message': 'Invalid transaction/signature'}
        return jsonify(response), 406
    else:
        response = {'message': 'Transaction will be added to the Block ' + str(transaction_results)}
        return jsonify(response), 201


@app.route('/access_transaction', methods=["POST"])
def access_transaction():
    values = request.form
    other_params = {'user_attribute': values['confirmation_user_public_key']}
    transaction_results = blockchain.submit_transaction(values['confirmation_user_public_key'],
                                                        "",
                                                        "",
                                                        other_params)
    if transaction_results == False:
        response = {'message': 'Invalid transaction/signature'}
        return jsonify(response), 406
    else:
        response = {'message': 'Transaction will be added to the Block ' + str(transaction_results)}
        return jsonify(response), 201


@app.route('/write_transaction', methods=["POST"])
def write_transaction():
    values = request.form
    other_params = {'ct': values['ct'], 'hash_ct': values['hash_ct'], 'Acc': values['confirmation_Acc']}
    transaction_results, link = blockchain.submit_transaction(values['confirmation_sender_public_key'],
                                                        values['transaction_signature'],
                                                        "",
                                                        other_params)
    if transaction_results == False:
        response = {'message': 'Invalid transaction/signature'}
        return jsonify(response), 406
    else:
        response = {'message': 'Transaction will be added to the Block ' + str(transaction_results),'url':link}
        return jsonify(response), 201


@app.route('/read_transaction', methods=["POST"])
def read_transaction():
    values = request.form
    other_params = {'hash_ct': values['hash_ct'], 'accept_reject': values['accept_reject']}
    transaction_results, link = blockchain.submit_transaction(values['confirmation_reader_public_key'],
                                                        values['transaction_signature'],
                                                        values['confirmation_timestamp'],
                                                        other_params)
    response = {'message': 'Transaction will be added to the Block ' + str(transaction_results), 'url':link}
    return jsonify(response), 201


@app.route('/nodes/get', methods=['GET'])
def get_nodes():
    nodes = list(blockchain.nodes)
    response = {'nodes': nodes}
    return jsonify(response), 200


@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.form
    nodes = values.get('nodes').replace(' ', '').split(',')
    if nodes is None:
        return "Error, please supply a a valid list of nodes", 400
    for node in nodes:
        blockchain.register_node(node)
    response = {
        'message' : 'nodes have been added',
        'total_nodes' : [node for node in blockchain.nodes]
    }
    return jsonify(response), 200


@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = blockchain.resolving_conflict()
    print(replaced)
    if replaced:
        response = {
            'message': 'our chain is replaced',
            'new_chain': blockchain.chain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': blockchain.chain
        }
    return jsonify(response), 200

@app.route('/storage/log', methods=['POST'])
def log_transaction():
    values = request.form
    log = blockchain.log(values['rid'],values['user_public_key'],values['operation'])


    response = {
        'message': 'Log completed',
        'path': './logs/' + str(blockchain.logs) + '.txt'
    }

    return jsonify(response), 200

#Authorization Smart Contract
@app.route('/verify_access', methods=["POST"])
def access_transaction():
    values = request.form
    other_params = {'user_attribute': values['confirmation_user_public_key']}
    operation = 1
    if values['operation'] == 'write':
        operation: 0
    rid = values['rid']
    transaction_results = decision_engine(values['uid'], values['rid'], operation)
    if transaction_results == False:
        
        #### Storage Communication
        blockchain.storage_request(rid, values['confirmation_user_public_key'])

        

        response = {'message': 'Access Granted'}
        return jsonify(response), 406
    else:
        response = {'message': f'Access Denied to resource of id :{rid}'}
        return jsonify(response), 201



def decision_engine(uid, rid, operation):
    config = OrderedDict([
    ('uid', uid),
    ('rid', rid),
    ('operation', operation),
    ('outdir', 'neural_network'),
    ('data', 'dataset/u4k-r4k-auth11k.sample'),
    ('debug', False),
    ('model_name', 'dlbac_alpha.hdf5'),
    ])
    model_path = '/home/sina/dev/blockchain/a_j.project/codes/blockchain/dlbac_alpha.hdf5'
    dlbac_alpha = load_model(model_path)
    x_tuple = data_parser(config)
    dlbac_alpha_probs = dlbac_alpha.predict(x_tuple)
    if debug:
        print('Output for given uid and rid', dlbac_alpha_probs)
    
    decision = (dlbac_alpha_probs > 0.5).astype(int)
    if debug:
        print('Output of the decision engine', decision)

    if decision[0, operation]:
        return True
    else:
        return False

def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def data_parser(config):
    id_count = 2 # uid and rid
    ops_count = 4
    metadata_count = 8 #in our experiment dataset (u4k-r4k-auth11k), each user/ resource has eight metadata
    
    debug = config['debug']
    dataFileName = config['data']
    uid = config['uid']
    rid = config['rid']

    cols = id_count + (metadata_count * 2) + ops_count # <uid rid><8 user-meta and 8 resource-meta><4 ops>

    # load the dataset
    raw_dataset = loadtxt(dataFileName, delimiter=' ', dtype=str)
    
    tuples_count = raw_dataset.shape[0]

    found = False
    # get index based on uid and rid
    for tuple_index in range(tuples_count):
        if raw_dataset[tuple_index, 0] == uid and raw_dataset[tuple_index, 1] == rid:
            found = True
            if debug:
                print('The user and resource pair found in index: %d\n' % (tuple_index))
            break
    if not found:
        if debug:
            logger.info('The user and resource pair doesn`t exist in the dataset.')
        exit(0)

    dataset = raw_dataset[:,2:cols] # TO SKIP UID RID

    feature = dataset.shape[1]
    if debug:
        print('Features:', feature)
    metadata = feature - ops_count

    urp = dataset[:,0:metadata]

    urp = to_categorical(urp)
    if debug:
        print('shape of URP after encoding')
        print(urp.shape)

    x_tuple = urp[tuple_index]
    if debug:
        print('x_tuple shape:', x_tuple.shape)
    x_tuple = x_tuple[..., np.newaxis]
    if debug:
        print('x_tuple after adding new shape:', x_tuple.shape)

    x_tuple = x_tuple[np.newaxis, ...]
    if debug:
        print('x_tuple after adding new shape:', x_tuple.shape)

    return x_tuple







if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5002, type=int, help="port to listen to")
    args = parser.parse_args()
    port = args.port
    app.run(host='127.0.0.1', port=port, debug=True)