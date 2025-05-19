from charm.toolbox.pairinggroup import PairingGroup
from charm.core.engine.util import bytesToObject

import io
from .pebel.cpabe import cpabe_setup, cpabe_keygen,cpabe_encrypt, cpabe_decrypt
from .pebel.util import write_key_to_file, read_key_from_file
from .pebel.exceptions import PebelDecryptionException


def setup(mpk_path, msk_path):
    """
    mpk_path: path to save the public key
    msk_path: path to save master key
    """
    group = PairingGroup('SS512')
    (mpk, msk) = cpabe_setup(group)
    write_key_to_file(mpk_path, mpk, group)
    write_key_to_file(msk_path, msk, group)
    with open(mpk_path, 'r') as f:
        public_key = f.read().replace('\n', '')
    with open(msk_path, 'r') as f:
        master_key = f.read().replace('\n', '')        
    return public_key, master_key


def keygen(mpk_path, msk_path,attributes, key_path):
    """
    msk_path: master key saved file path
    mpk_path: public key saved file path
    attributes: attribute of user
    key_path: path to save the key for user
    """
    group = PairingGroup('SS512')
    mpk = read_key_from_file(mpk_path, group)
    msk = read_key_from_file(msk_path, group)
    dec_key = cpabe_keygen(group, msk, mpk, attributes)
    write_key_to_file(key_path, dec_key, group)
    with open(key_path, 'r') as f:
        key = f.read().replace('\n', '')
    return key


def encrypt(mpk_path, plain_text_file, policy):
    group = PairingGroup('SS512')
    mpk = read_key_from_file(mpk_path, group)
    ctxt = cpabe_encrypt(group, mpk, io.open(plain_text_file,'rb'), policy)
    ctxt_fname = "".join([plain_text_file, ".cpabe"])
    with io.open(ctxt_fname, 'wb') as ctxt_file:
        for b in ctxt:
            ctxt_file.write(bytes([b]))


def decrypt(cipher_text_file, mpk_path, key):
    group = PairingGroup('SS512')
    key = bytes(key.encode('utf8'))
    key = bytesToObject(key, group)
    ptxt_fname = cipher_text_file.replace(".cpabe", ".prime")
    mpk = read_key_from_file(mpk_path, group)
    # key = read_key_from_file(key_path, group)
    result = ''
    try:
        raw = cpabe_decrypt(group, mpk, key, io.open(cipher_text_file, 'rb'))
    except PebelDecryptionException as e:
        print("Unable to decrypt ciphertext: {}".format(e))
    else:
        with io.open(ptxt_fname, 'wb') as ptxt:
            for b in raw:
                ptxt.write(bytes([b]))
                ptxt.flush()
                result += (bytes([b])).decode("utf-8") 
        return  result

