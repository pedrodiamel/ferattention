
import os
import sys
import shutil
import numpy as np
import csv


from tqdm import tqdm

from itertools import islice
from PIL import Image

from pytvision.datasets.imageutl import dataProvide
from pytvision.transforms.rectutils import Rect

from .utility import download_url, check_integrity

train = ['FER2013Train']
valid = ['FER2013Valid']
test = ['FER2013Test']

# List of folders for training, validation and test.
folder_names = {'Training'   : 'FER2013Train',
                'PublicTest' : 'FER2013Valid',
                'PrivateTest': 'FER2013Test'}

def str_to_image(image_blob):
    ''' Convert a string blob to an image object. '''
    image_string = image_blob.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    return Image.fromarray(image_data)

def generate_training_data(base_folder, fer_path, ferplus_path):
    '''
    Generate PNG image files from the combined fer2013.csv and fer2013new.csv file. The generated files
    are stored in their corresponding folder for the trainer to use.
    
    Args:
        base_folder(str): The base folder that contains  'FER2013Train', 'FER2013Valid' and 'FER2013Test'
                          subfolder.
        fer_path(str): The full path of fer2013.csv file.
        ferplus_path(str): The full path of fer2013new.csv file.
    '''
    
    print("Start generating ferplus images.")
    
    for key, value in folder_names.items():
        folder_path = os.path.join(base_folder, value)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    ferplus_entries = []
    with open(ferplus_path,'r') as csvfile:
        ferplus_rows = csv.reader(csvfile, delimiter=',')
        for row in tqdm( islice(ferplus_rows, 1, None) ):
            ferplus_entries.append(row)
 
    index = 0
    with open(fer_path,'r') as csvfile:
        fer_rows = csv.reader(csvfile, delimiter=',')
        for row in tqdm( islice(fer_rows, 1, None) ):
            ferplus_row = ferplus_entries[index]
            file_name = ferplus_row[1].strip()
            if len(file_name) > 0:
                image = str_to_image(row[1])
                image_path = os.path.join(base_folder, folder_names[row[2]], file_name)
                image.save(image_path, compress_level=0)     
                #print(image_path)           
            index += 1 
            
    print("generate ferp dataset done...")


class FERPDataset( dataProvide ):
    """
    FER PLUS dataset
    A custom reader for FER+ dataset that support multiple modes as described in:
        https://arxiv.org/abs/1608.01041
    """

    git_fer='https://github.com/Microsoft/FERPlus.git'
    file_fer='fer2013.csv'
    file_fernew='fer2013new.csv'
    base_folder='fer2013'
    url = 'https://www.dropbox.com/s/03ecqa15qe2dqzc/fer2013.tar?dl=0'
    filename='fer2013.tar'
    label_file_name='label.csv'

    # Emotions class
    # Neutral - NE, Happiness - HA, Surprise - SU, Sadness - SA, Anger - AN, Disgust - DI, Fear - FR, Contempt - CO
    classes = ['Neutral - NE', 'Happiness - HA', 'Surprise - SU', 'Sadness - SA', 'Anger - AN', 'Disgust - DI', 'Fear - FR', 'Contempt - CO']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}
    

    def __init__(self, 
        root, 
        sub_folders, 
        shuffle=False,
        transform=None,
        download=False,
        ):
        """
        Each sub_folder contains the image files and a csv file for the corresponding label. The read iterate through
        all the sub_folders and aggregate all the images and their corresponding labels. 
        Args:
            root (str): The base folder that contains  'FER2013Train', 'FER2013Valid' and 'FER2013Test' subfolder.                          
            sub_folders (str): 
            ...
            transform (callable, optional): Optional transform to be applied on a sample.                
        """      

        self.root = os.path.expanduser( root )
        self.sub_folders     = sub_folders
        self.emotion_count   = 8
        self.shuffle         = shuffle

        if download:
            self.download()
        
        self.load_folders()
        self.labels =  np.array( [ np.argmax(self.data[i][1]) for i in range( len(self.data) )  ])

        # not used
        self.training_mode   = 'majority' 
        self.transform = transform     


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   
        #check index
        if idx<0 and idx>len(self.data): 
            raise ValueError('Index outside range')
        self.index = idx
        pathname = self.data[idx][0]
        image = np.array(self._loadimage(pathname), dtype=np.uint8)
        label = np.array(self.data[idx][1], dtype=np.float32)
        label = np.argmax(label)
        return image, label

    def getroi(self):
        return self.data[self.index][2]

    def load_folders(self):
        '''
        Load the actual images from disk. While loading, we normalize the input data.
        '''
        
        self.data = []
        self.per_emotion_count = np.zeros(self.emotion_count, dtype=np.int)
        
        for folder_name in self.sub_folders: 
            #logging.info("Loading %s" % (os.path.join(self.root, folder_name)))
            folder_path = os.path.join(self.root, folder_name)
            in_label_path = os.path.join(folder_path, self.label_file_name)
            with open(in_label_path) as csvfile: 
                emotion_label = csv.reader(csvfile) 
                for row in emotion_label: 
                    
                    # load the image
                    image_path = os.path.join(folder_path, row[0])
                    # face rectangle 
                    box = list(map(int, row[1][1:-1].split(',')))
                    face_rc = Rect(box)

                    emotion_raw = list(map(float, row[2:len(row)]))
                    emotion = self._process_data(emotion_raw)

                    idx = np.argmax(emotion)
                    if idx < self.emotion_count: # not unknown or non-face 
                        emotion = emotion[:-2]
                        emotion = [float(i)/sum(emotion) for i in emotion]
                        self.data.append((image_path, emotion, face_rc))
                        self.per_emotion_count[idx] += 1
        
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _process_target(self, target):
        '''
        Based on https://arxiv.org/abs/1608.01041 the target depend on the training mode.

        Majority or crossentropy: return the probability distribution generated by "_process_data"
        Probability: pick one emotion based on the probability distribtuion.
        Multi-target: 
        '''
        if self.training_mode == 'majority' or self.training_mode == 'crossentropy': 
            return target
        elif self.training_mode == 'probability': 
            idx             = np.random.choice(len(target), p=target) 
            new_target      = np.zeros_like(target)
            new_target[idx] = 1.0
            return new_target
        elif self.training_mode == 'multi_target': 
            new_target = np.array(target) 
            new_target[new_target>0] = 1.0
            epsilon = 0.001     # add small epsilon in order to avoid ill-conditioned computation
            return (1-epsilon)*new_target + epsilon*np.ones_like(target)

    def _process_data(self, emotion_raw):
        '''
        Based on https://arxiv.org/abs/1608.01041, we process the data differently depend on the training mode:

        Majority: return the emotion that has the majority vote, or unknown if the count is too little.
        Probability or Crossentropty: convert the count into probability distribution.abs
        Multi-target: treat all emotion with 30% or more votes as equal.
        '''        

        mode = 'majority'

        size = len(emotion_raw)
        emotion_unknown     = [0.0] * size
        emotion_unknown[-2] = 1.0

        # remove emotions with a single vote (outlier removal) 
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size 

        if mode == 'majority': 
            # find the peak value of the emo_raw list 
            maxval = max(emotion_raw) 
            if maxval > 0.5*sum_list: 
                emotion[np.argmax(emotion_raw)] = maxval 
            else: 
                emotion = emotion_unknown   # force setting as unknown 
        elif (mode == 'probability') or (mode == 'crossentropy'):
            sum_part = 0
            count = 0
            valid_emotion = True
            while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
                maxval = max(emotion_raw) 
                for i in range(size): 
                    if emotion_raw[i] == maxval: 
                        emotion[i] = maxval
                        emotion_raw[i] = 0
                        sum_part += emotion[i]
                        count += 1
                        if i >= 8:  # unknown or non-face share same number of max votes 
                            valid_emotion = False
                            if sum(emotion) > maxval:   # there have been other emotions ahead of unknown or non-face
                                emotion[i] = 0
                                count -= 1
                            break
            if sum(emotion) <= 0.5*sum_list or count > 3: # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
                emotion = emotion_unknown   # force setting as unknown 
        elif mode == 'multi_target':
            threshold = 0.3
            for i in range(size): 
                if emotion_raw[i] >= threshold*sum_list: 
                    emotion[i] = emotion_raw[i] 
            if sum(emotion) <= 0.5 * sum_list: # less than 50% of the votes are integrated, we discard this example 
                emotion = emotion_unknown   # set as unknown 
                                
        return [float(i)/sum(emotion) for i in emotion]

    def download(self):
        import tarfile
        
        if os.path.exists(  os.path.join(self.root, self.base_folder) ):
            return 
    
        root = self.root
        fer_path = os.path.join(root, self.base_folder, self.file_fer)
        ferplus_path = os.path.join(root, self.file_fernew)       

        #wget      
        os.system('wget --output-document={} {}'.format( os.path.join(root, self.filename), self.url ) )

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)        

        try:
            os.system( 'git clone {} {}'.format( self.git_fer, 'tmp' ) )
            shutil.copy( 'tmp/fer2013new.csv',  os.path.join( self.root, 'fer2013new.csv' )  )
            for key, value in tqdm( folder_names.items() ):
                folder_path_src = os.path.join('tmp/data', value, 'label.csv' )
                folder_path_dest = os.path.join(root, value )
                if not os.path.exists(folder_path_dest):
                    os.mkdir(folder_path_dest)
                folder_path_dest = os.path.join(folder_path_dest, 'label.csv' )                
                shutil.copy( folder_path_src, folder_path_dest)
            shutil.rmtree('tmp')
        except:            
            assert(False)

        generate_training_data( root, fer_path, ferplus_path)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        return fmt_str




