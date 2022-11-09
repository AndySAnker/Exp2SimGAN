"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
import os.path, random, pdb
import numpy as np
from data.image_folder import make_dataset
import util.util as util
import torch
        
# from PIL import Image
def rgb2gray(rgb): return 0.2989 * rgb[0,:,:] + 0.5870 * rgb[1,:,:] + 0.1140 * rgb[2,:,:]  


class Unaligned3DDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=99999, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """

        print ("\n >>> WARNING: You are using the 3D model loader, which does not have any functionalities regarding preprocessing the data. Therefore, you must do this before loading the data. <<< \n")

        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B


        # get the image paths of your dataset;
        #self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        #self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """

        #path = 'temp'    # needs to be a string
        #data_A = None    # needs to be a tensor
        #data_B = None    # needs to be a tensor
        #return {'data_A': data_A, 'data_B': data_B, 'path': path}

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = np.load(A_path)
        B_img = np.load(B_path)

        A_img = torch.tensor(A_img).float()
        B_img = torch.tensor(B_img).float()

        # Has to change get_transform to get to work
#        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
#        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
#        transform = get_transform(modified_opt)

#        A = transform(A_img)
#        B = transform(B_img)

        A = A_img
        B = B_img

        # Make back to greyscale
        if self.opt.input_nc == 1 and self.opt.output_nc == 1 and A.ndim == 4:
            A = rgb2gray(A).unsqueeze(0)
            B = rgb2gray(B).unsqueeze(0)
        if A.ndim < 4:
            A = (A).unsqueeze(0)
            B = (B).unsqueeze(0)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        
    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)
