from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import datetime
from collections import OrderedDict
from CPABE.CPABE import setup, encrypt, decrypt
from Crypto import Hash
from pathlib import Path
import hashlib
import pickle


file_ACC = dict()
def save_file_Acc(file_name, ACC, f_hash):
    global file_ACC
    file_ACC[file_name] = [ACC, f_hash]
    # save as pickle
    with open('../keys/file_acc_hash.pickle', 'wb') as handle:
        pickle.dump(file_ACC, handle, protocol=pickle.HIGHEST_PROTOCOL)

class Write_Transaction:
    def __init__(self, sender_public_key, sender_private_key, file_name, Acc):
        self.file_name = file_name
        self.Acc = Acc
        self.Acc = self.edit_acc(self.Acc)
        self.sender_public_key = sender_public_key
        self.sender_private_key = sender_private_key

    def edit_acc(self, Acc):
        new_acc = Acc.split(' ')
        acc = ''
        for i in new_acc:
            if i != 'and' and i != 'or':
                acc += i.upper() + ' '
            else:
                acc += i + ' '
        return acc

    def Encrypt(self):
        encrypt('../keys/system.mpk', f'../files/{self.file_name}', self.Acc)
        filepath = f'../files/{self.file_name}.cpabe'
        ct = Path(filepath).read_bytes()
        ct_hash = hashlib.sha1(ct)
        return ct, ct_hash

    def to_dict(self):
        ct, ct_hash = self.Encrypt()
        return OrderedDict({
            'sender_public_key': str(self.sender_public_key,),
            'ct': str(ct),
            'hash_ct': str(ct_hash.hexdigest()),
            'Acc': str(self.Acc)
        })

    def sign_transaction(self, transaction):
        # use specific formula with self.sender_private_key to generate signed data self.to_dict
        h = SHA.new(str(transaction).encode('utf8'))
        str_hex = str(h.hexdigest())
        h = SHA.new(self.sender_private_key.encode('utf8'))
        str_hex1 = str(h.hexdigest())
        # return signed to_dict
        sign = str_hex + str_hex1
        return sign


class Read_Transaction:
    def __init__(self, reader_public_key, reader_key, file_name):
        self.timestamp = datetime.datetime.now().timestamp()
        self.reader_public_key = reader_public_key
        self.reader_key = reader_key
        self.file_name = file_name
        with open('../keys/file_acc_hash.pickle', 'rb') as handle:
            file_acc_hash = pickle.load(handle)
        self.file_hash = file_acc_hash[self.file_name[:-6]][1]
        self.dec_result = self.Decrypt()

    def Decrypt(self):
        res = decrypt(f'../files/{self.file_name}', '../keys/system.mpk', self.reader_key)
        if res != None:
            return True
        else:
            return False

    def sign_transaction(self, transaction):
        # sign with specific formula self.to_dict()
        h = SHA.new(str(transaction).encode('utf8'))
        str_hex = str(h.hexdigest())
        h = SHA.new(self.reader_key.encode('utf8'))
        str_hex1 = str(h.hexdigest())
        # return signed to_dict
        sign = str_hex + str_hex1
        return sign

    def to_dict(self):
        return OrderedDict({
            'reader_public_key': str(self.reader_public_key),
            'file_hash': self.file_hash,
            'timestamp': str(self.timestamp,),
            'accept_reject': str(self.dec_result).lower()
        })


app = Flask(__name__)
CORS(app)

count = 0

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/write')
def write_request():
    return render_template('write.html')


@app.route('/read')
def read_request():
    return render_template('read.html')


@app.route('/create_key')
def add_user():
    global count
    count += 1
    public_key, private_key = setup(f'../keys/{count}.mpk', f'../keys/{count}.msk')
    response = {
        'private_key': private_key,
        'public_key': public_key
    }
    return jsonify(response), 201


@app.route('/write/new', methods=['POST'])
def create_write_request():
    sender_public_key = request.form['sender_public_key']
    sender_private_key = request.form['sender_private_key']
    file_name = request.form['file_name']
    Acc = request.form['acc_rule']
    w_transaction = Write_Transaction(sender_public_key, sender_private_key, file_name, Acc)
    transaction = w_transaction.to_dict()
    response = {'transaction': transaction,
                'signature': w_transaction.sign_transaction(transaction)}
    f_hash = response['transaction']['hash_ct']
    save_file_Acc(file_name, Acc, f_hash)
    return jsonify(response), 200


@app.route('/read/new', methods=['POST'])
def create_read_request():
    reader_public_key = request.form['reader_public_key']
    reader_key = request.form['key']
    file_name = request.form['file_name']
    r_transaction = Read_Transaction(reader_public_key, reader_key, file_name)
    transaction = r_transaction.to_dict()
    response = {'transaction': transaction,
                'signature': r_transaction.sign_transaction(transaction)}
    return jsonify(response), 200


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=8081, type=int, help="port to listen to")
    args = parser.parse_args()
    port = args.port
    app.run(host='127.0.0.1', port=port, debug=True)
