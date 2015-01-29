# Cooccurrence matrix construction tools
# for fitting the GloVe model.
import numpy as np
import scipy.sparse as sp 

USE_HICKLE = True
try:
    import h5py
    import hickle
except:
    print "failed to load hickle, switching back to pickle"
    USE_HICKLE = False
    try:
        # Python 2 compat
        import cPickle as pickle
    except ImportError:
        import pickle


from .corpus_cython import construct_cooccurrence_matrix

class sparseMat(object):
    def __init__(self, matrix, dictionary):
        self.shape = matrix.shape
        self.col = matrix.col
        self.row = matrix.row
        self.data = matrix.data
        self.dictionary = dictionary

    def toMatrix():
        return sp.coo_matrix((self.data, (self.row, self.col)),
                             shape=self.shape,
                             dtype=np.float64)
    def dictionary():
        return self.dictonary

class Corpus(object):
    """
    Class for constructing a cooccurrence matrix
    from a corpus.

    A dictionry mapping words to ids can optionally
    be supplied. If left None, it will be constructed
    from the corpus.
    """
    
    def __init__(self, dictionary=None):

        self.dictionary = {}
        self.dictionary_supplied = False
        self.matrix = None

        if dictionary is not None:
            self._check_dict(dictionary)
            self.dictionary = dictionary
            self.dictionary_supplied = True

    def _check_dict(self, dictionary):

        if (np.max(list(dictionary.values())) != (len(dictionary) - 1)):
            raise Exception('The largest id in the dictionary '
                            'should be equal to its length minus one.')

        if np.min(list(dictionary.values())) != 0:
            raise Exception('Dictionary ids should start at zero')

    def fit(self, corpus, window=10, max_map_size=1000, ignore_missing=False):
        """
        Perform a pass through the corpus to construct
        the cooccurrence matrix. 

        Parameters:
        - iterable of lists of strings corpus
        - int window: the length of the (symmetric)
          context window used for cooccurrence.
        - int max_map_size: the maximum size of map-based row storage.
                            When exceeded a row will be converted to
                            more efficient array storage. Setting this
                            to a higher value will increase speed at
                            the expense of higher memory usage.
        - bool ignore_missing: whether to ignore words missing from
                               the dictionary (if it was supplied).
                               Context window distances will be preserved
                               even if out-of-vocabulary words are
                               ignored.
                               If False, a KeyError is raised.
        """
        
        self.matrix = construct_cooccurrence_matrix(corpus,
                                                    self.dictionary,
                                                    int(self.dictionary_supplied),
                                                    int(window),
                                                    int(ignore_missing),
                                                    max_map_size)

    def save(self, filename):
        if USE_HICKLE:
            f = h5py.File(filename, "w")
            f.create_dataset('matrix_shape', data=self.matrix.shape)
            f.create_dataset('matrix_col', data=self.matrix.col)
            f.create_dataset('matrix_row', data=self.matrix.row)
            f.create_dataset('matrix_data', data=self.matrix.data)
            f.create_dataset('dic_keys',data=self.dictionary.keys())
            f.create_dataset('dic_values',data=self.dictionary.values())
            f.close()
            #print type(self.dictionary)
            #print len(self.dictionary)
            #hickle.dump(self.dictionary, filename, mode='w')
            #dic = {'data':self.matrix.data, 'row':self.matrix.row, 'col':self.matrix.col, 'shape':self.matrix.shape, 'dictionary':self.dictionary}
            #for key in dic:
            #    hickle.dump(dic[key], filename+"_"+key, mode='w') #, compression='gzip')
        else:
            with open(filename, 'wb') as savefile:
                pickle.dump((self.dictionary, self.matrix),
                            savefile,
                            protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):

        instance = cls()

        if USE_HICKLE:
            f = h5py.File(filename,'r')
            m_shape = f['matrix_shape'][:]
            m_col = f['matrix_col'][:]
            m_row = f['matrix_row'][:]
            m_data = f['matrix_data'][:]
            instance.matrix = sp.coo_matrix((m_data, (m_row, m_col)),
                             shape=m_shape,
                             dtype=np.float64)
            keys, values =  f['dic_keys'][:], f['dic_values'][:]
            dic = {}
            for k,v in zip(keys,values):
                dic.setdefault(k,v)
            instance.dictionary = dic
        else:
            with open(filename, 'rb') as savefile:
                instance.dictionary, instance.matrix = pickle.load(savefile)             

        return instance
