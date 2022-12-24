import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd
import struct
import gzip

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
          # axis = W i.e. 1
          return np.flip(img, 1)
        else:
          return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        h,w,c = img.shape
        padded_img = np.pad(img, ((self.padding, self.padding),(self.padding, self.padding),(0,0)), 'constant', constant_values=0)
        shift_x += self.padding
        shift_y += self.padding
        return padded_img[shift_x: shift_x+h, shift_y: shift_y+w,:]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        dtype = "float32",
        device = nd.cpu()
    ):
        self.dtype = dtype
        self.device = device
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.batch_idx = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.batch_idx == 0 and self.shuffle:
            arr = np.arange(len(self.dataset))
            np.random.shuffle(arr)
            self.ordering = np.array_split(arr,
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        if self.batch_idx == len(self.ordering):
            raise StopIteration
        # batch = self.dataset[self.ordering[self.batch_idx]]
        batch = [self.dataset[i] for i in self.ordering[self.batch_idx]]

        result = [list() for i in range(len(batch[0]))]
        for b in batch:
            for i, data in enumerate(b):
                result[i].append(data[0] if len(data.shape) == 1 and data.shape[0] == 1 else data)
        self.batch_idx += 1
        return tuple([Tensor(np.array(l), require_grad=False, device=self.device, dtype=self.dtype) for l in result])
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.root = (image_filename, label_filename)
        self.transforms = transforms
        self.data, self.targets = self.parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        trans_img = self.data[index]
        if self.transforms is not None:
          for transform in self.transforms:
            trans_img = transform(trans_img)
        return (trans_img, self.targets[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.targets)
        ### END YOUR SOLUTION
    @staticmethod
    def parse_mnist(image_filename, label_filename):
      """ Read an images and labels file in MNIST format.  See this page:
      http://yann.lecun.com/exdb/mnist/ for a description of the file format.

      Args:
          image_filename (str): name of gzipped images file in MNIST format
          label_filename (str): name of gzipped labels file in MNIST format

      Returns:
          Tuple (X,y):
              X (numpy.ndarray[np.float32]): 4D numpy array containing the loaded
                  data.  The dimensionality of the data should be
                  (num_examples x 28 x 28 x 1) Values should be of type np.float32, and the data
                  should be normalized to have a minimum value of 0.0 and a
                  maximum value of 1.0.

              y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                  labels of the examples.  Values should be of type np.int8 and
                  for MNIST will contain the values 0-9.
      """
      ### BEGIN YOUR CODE
      with gzip.open(label_filename,'rb') as f_lb:
        # All the integers in the files are stored in the MSB first (high endian) format.
        # Users of Intel processors and other low-endian machines must flip the bytes of the header.
        # > donotes big-endian(aka high-endian); I denotes a 32-bit integer
        # read the first two 32-bit intergers describing data type and number of labels.
        # as we've already know that the offset of first label is 8, it means that
        # the first 8 bytes are magic number and number of labels.
        magic_lb, num_lb = struct.unpack('>II', f_lb.read(4*2))
        # read the rest of data (num_lb labels)
        labels = np.frombuffer(f_lb.read(), dtype = np.uint8)
      with gzip.open(image_filename, 'rb') as f_img:
        # the difference of image data compared with label data is that there're four 32-bit integers
        # describing info (datatype, number, number of rows, number of cols), and every
        # 8-bit unsigned byte is not a label but a pixel.
        magic_img, num_img, num_rows, num_cols = struct.unpack('>IIII', f_img.read(4*4))
        images = np.frombuffer(f_img.read(),dtype=np.ubyte).reshape(num_img, num_rows*num_cols).reshape((-1, 28, 28, 1))
        # normalize the images
        norm_images = images.astype(np.float32) / (np.amax(images).astype(np.float32)-np.amin(images).astype(np.float32))
      return norm_images, labels
      ### END YOUR CODE


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        def unpickle(file):
            import pickle
            # print("filename:", file)
            with open(file, 'rb') as fo:
                dic = pickle.load(fo, encoding='bytes')
            return dic

        # Assume that CIFAR-10 dataset has been downloaded.
        if base_folder is None:
            assert os.path.isdir("./data/cifar-10-batches-py")
            base_folder = "./data/cifar-10-batches-py"
        if train:
            dic = unpickle(os.path.join(base_folder, "data_batch_1"))
            for i in range(2,6):
                d = unpickle(os.path.join(base_folder, "data_batch_" + str(i)))
                dic[b"data"] = np.concatenate((dic[b"data"], d[b"data"]), axis=0)
                dic[b"labels"] += d[b"labels"]
        else:
            dic = unpickle(os.path.join(base_folder,"test_batch"))
        self.X = (dic[b"data"]/255).astype("float32")
        self.y = np.array(dic[b"labels"])
        assert self.X.shape[0] == len(self.y)
        self.transforms = transforms
        ### END YOUR SOLUTION


    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        assert self.X[index].size == 3*32*32
        X_idx = self.X[index].reshape((3, 32, 32))
        if self.transforms is not None:
            for transform in self.transforms:
                X_idx = transform(X_idx)
        return X_idx, self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        import re
        with open(path, 'r') as f:
            lines = f.readlines()
        if max_lines and len(lines)>max_lines:
            lines = lines[:max_lines]
        # return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
        lines = [line+'<eos>' for line in lines]
        ids = [self.dictionary.add_word(word) for line in lines for word in line.split()]
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e.g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    # data: a list of ids
    assert len(data)>=batch_size
    if len(data)%batch_size>0:
        end = len(data)-(len(data)%batch_size)
        data = data[:end]
    return np.array(data).reshape((batch_size, -1)).transpose()
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    nbatch, bs = batches.shape
    seq_len = min(bptt, nbatch-1-i)
    data = Tensor(batches[i:i+seq_len, :], device=device, dtype=dtype)
    target = Tensor(batches[i+1:i+1+seq_len, :].reshape(seq_len*bs), device=device, dtype=dtype)
    return data, target
    ### END YOUR SOLUTION