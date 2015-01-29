from glove import Glove, Corpus

inputFile = "/media/charles/data/nlp/zzz1000"
corpusModelFile = "/media/charles/data/nlp/corpus_wiki.model"
outputFile = "/media/charles/data/nlp/glove_wiki.model"
epochs = 10
nb_threads = 4


def get_text(fin):
    f = open(fin)     
    for line in f:    
        yield line[:-1].split(' ')
        
#corpus_model = Corpus()  
#print("computing coocurrence matrix...")       
#corpus_model.fit(get_text(inputFile), window=10)
#print("saving coocurrence matrix...")
#corpus_model.save(corpusModelFile)
corpus_model = Corpus.load(corpusModelFile)
print("fitting model...")
glove = Glove(no_components=200, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=epochs,
                  no_threads=nb_threads, verbose=True)

glove.add_dictionary(corpus_model.dictionary)
print("saving model to "+outputFile+" ...")
glove.save(outputFile)
