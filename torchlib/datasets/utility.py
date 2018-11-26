

import os
import os.path
import sys
import hashlib
import errno

import multiprocessing 
import urllib3
import csv

from PIL import Image
from io import BytesIO

import numpy as np
import cv2
import random
import requests

from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36',
}

def download_images( pack ):
    '''
    @pack: [ key, url, output ]
    '''
    
    (key, url, out_dir) = pack 

    filename = os.path.join(out_dir, '{}.jpg'.format( key ) )
    if os.path.exists(filename):
        print( 'Image {} already exists. Skipping download.'.format( filename ) )
        return

    try:
        #print('Trying to get %s.' % url)
        http = urllib3.PoolManager()
        response = http.request('GET', url )
        image_data = response.data

        # response = requests.get(d['url'][0], allow_redirects=True, timeout=60, headers=headers)
        # if r.status_code != 200:
        #     print('status code != 200', response.status_code, url )
        #     return
        # image_data = response.content

    except:
        print( 'Warning: Could not download image {} from {}'.format(key, url)  )
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {} {}'.format(key,url))
        return
    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key) )
        return
    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {}'.format(filename) )
        return

def gen_bar_updator(pbar):
    def bar_update(count, block_size, total_size):
        pbar.total = total_size / block_size
        pbar.update(count)

    return bar_update

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)

    print(md5o.hexdigest())
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True

def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updator(tqdm()))
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files