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


def hash_func(transactions):
    data = str(transactions)
    hash_data = sha256(data.encode('utf-8')).hexdigest()
    return hash_data


class DecisionEngine:
    def __init__(self):
        self.chain = []
        self.transactions = []
        self.node_id = str(uuid4()).replace('-', '')
        self.nodes = set()
        # add genesis block to blockchain
        self.genesis()


blockchain = DecisionEngine()
app = Flask(__name__)
CORS(app)

@app.route('/verify_access_transaction', methods=["POST"])
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





if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Storage always runs on this port:
    parser.add_argument('-p', '--port', default=3000, type=int, help="port to listen to")
    args = parser.parse_args()
    port = args.port
    app.run(host='127.0.0.1', port=port, debug=True)




def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--uid', type=str, required=True)
    parser.add_argument('--rid', type=str, required=True)
    parser.add_argument('--operation', type=str, required=True)
    parser.add_argument('--data', type=str, default='dataset/u4k-r4k-auth11k.sample')
    parser.add_argument('--debug', type=str2bool, default=False)

    args = parser.parse_args()

    config = OrderedDict([
        ('uid', args.uid),
        ('rid', args.rid),
        ('operation', args.operation),
        ('outdir', 'neural_network'),
        ('data', args.data),
        ('debug', args.debug),
        ('model_name', 'dlbac_alpha.hdf5'),
    ])

    return config


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


def main():
    # parse command line arguments
    config = parse_args()

    debug = config['debug']
    requested_ops = int(config['operation'].lower().split('op')[1]) - 1
    if debug:
        print('requested operation:', config['operation'])
        print('requested operation index:', requested_ops)
        logger.info(json.dumps(config, indent=2))

    # create output directory
    outdir = config['outdir']
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    x_tuple = data_parser(config)
    if debug:
        print('x_tuple shape after return:', x_tuple.shape)
    
    model_path = os.path.join(outdir, config['model_name'])
    
    if os.path.exists(model_path):
        print('Loading trained model from {}.'.format(model_path))
        dlbac_alpha = load_model(model_path)
    else:
        print('No trained model found at {}.'.format(model_path))
        exit(0)

    dlbac_alpha_probs = dlbac_alpha.predict(x_tuple)
    if debug:
        print('Output for given uid and rid', dlbac_alpha_probs)
    
    decision = (dlbac_alpha_probs > 0.5).astype(int)
    if debug:
        print('Output of the decision engine', decision)

    if decision[0, requested_ops]:
        print('***********************************')
        print('******** Access Granted! **********')
        print('***********************************')
    else:
        print('***********************************')
        print('******** Access Denied! ***********')
        print('***********************************')


if __name__ == '__main__':
    main()

