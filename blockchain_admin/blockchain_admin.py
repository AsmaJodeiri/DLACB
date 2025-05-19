from flask import Flask, render_template, jsonify, request
from collections import OrderedDict
from flask_cors import CORS
from CPABE.CPABE import setup, keygen
from Crypto.Hash import SHA
import datetime
import pickle


class USER:
    def __init__(self):
        self.idd = 0
        self.all_user = []
        self.com_user = []
        self.not_complete = []
        self.system_public_key = ''
        self.system_master_key = ''
        self.public_and_sk = dict()

    def add_user(self, public_key):
        self.idd += 1
        idd = f'user{self.idd}'
        user_info = {'public_key': public_key,
                'id': idd,
                'confirm': 0}
        self.all_user.append(user_info)

    def not_complete_user_list(self):
        self.not_complete = []
        for i in self.all_user:
            if i['confirm'] == 0:
                self.not_complete.append(i)
        return self.not_complete

    def confirm_for_user(self, public_key):
        for i in self.all_user:
            if public_key == i['public_key']:
                i['confirm'] = 1

    def save_user_key(self, public_key, sk):
        self.public_and_sk[public_key] = sk
        with open('../keys/public_secret_key.pickle', 'wb') as handle:
            pickle.dump(self.public_and_sk, handle, protocol=pickle.HIGHEST_PROTOCOL)


class setup_transaction:
    def __init__(self, system_public_key, system_master_key, universal_attribute):
        self.timestamp = datetime.datetime.now().timestamp()
        self.system_public_key = system_public_key
        self.system_master_key = system_master_key
        self.universal_attribute = universal_attribute

    def sign_transaction(self, key, transaction):
        # sign with specific formula self.to_dict()
        h = SHA.new(str(transaction).encode('utf8'))
        str_hex = str(h.hexdigest())
        h = SHA.new(key.encode('utf8'))
        str_hex1 = str(h.hexdigest())
        # return signed to_dict
        sign = str_hex + str_hex1
        return sign

    def to_dict(self):
        return OrderedDict({
            'admin_public_key': self.system_public_key,
            'timestamp': str(self.timestamp),
            'universal_attribute': self.universal_attribute
        })


class Access_Transaction:
    def __init__(self, public_key, user_attribute):
        self.public_key = public_key
        self.user_attribute = user_attribute

    def to_dict(self):
        return OrderedDict({
            'public_key': self.public_key,
            'user_attribute': self.user_attribute})


user_count = 0
user = USER()
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/confirm_user')
def confirm_user():
    not_complete = user.not_complete_user_list()
    return render_template('confirm_user.html', not_complete=not_complete)



@app.route('/create_master_key', methods=['POST'])
def admin_master_key():
     # landa = SS512
    landa = request.form['landa']
    public_key, master_key = setup('../keys/system.mpk', f'../keys/system.msk', landa)
    response = {
        'master_key': master_key,
        'public_key': public_key
    }
    return jsonify(response), 200


@app.route('/system_setup', methods=["POST"])
def system_setup():
    system_public_key = request.form['admin_public_key']
    system_master_key = request.form['admin_private_key']
    user.system_master_key = system_master_key
    user.system_public_key = system_public_key
    universal_attribute = request.form['universal_attribute']
    uni_attribute = [str(i).upper() for i in universal_attribute.split(',')]
    key = keygen('../keys/system.mpk', '../keys/system.msk', uni_attribute, '../keys/system.key')
    with open('../keys/system.mpk', 'r') as f:
        public_key = f.read().replace('\n', '')
    user.save_user_key(public_key, key)
    s_transaction = setup_transaction(system_public_key, system_master_key, universal_attribute)
    transaction = s_transaction.to_dict()
    response = {'transaction': transaction,
                'signature': s_transaction.sign_transaction(key, transaction)}
    return jsonify(response), 200


@app.route('/confirm_user/add', methods=['POST'])
def add_user_to_blockchain():
    values = request.form
    required = ['confirmation_sender_public_key']
    if not all(k in values for k in required):
        return 'Missing values', 400
    else:
        response = {'message': 'Transaction will be added to the Block '}
        # save user public key
        user.add_user(values['confirmation_sender_public_key'])
        return jsonify(response), 201


@app.route('/user/secret_key', methods=['POST'])
def generate_secure_key():
    global user_count
    user_count += 1
    public_key = request.form['public_key']
    user_attribute = request.form['user_attribute']
    u_attribute = [str(i).upper() for i in user_attribute.split(',')]
    key = keygen('../keys/system.mpk', '../keys/system.msk', u_attribute, f'../keys/{user_count}.key')
    response = {'key': key}
    user.save_user_key(public_key, key)
    return jsonify(response), 201


@app.route('/user/access_transaction', methods=['POST'])
def transaction_access():
    user_public_key = request.form['public_key']
    user_attribute = request.form['user_attribute']
    u_attribute = [str(i).upper() for i in user_attribute.split(',')]
    access_transaction = Access_Transaction(user_public_key, u_attribute)
    access_transaction.to_dict()
    user.confirm_for_user(user_public_key)
    response = {'transaction': access_transaction.to_dict()}
    return jsonify(response), 201


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5001, type=int, help="port to listen to")
    args = parser.parse_args()
    port = args.port
    app.run(host='127.0.0.1', port=port, debug=True)
