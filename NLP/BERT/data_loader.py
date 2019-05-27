# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
        #    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.

def seek_random_offset(inputFile, back_margin=2000):
    """ seek random offset of file pointer """
    inputFile.seek(0, 2)
    # we remain some amount of text to read
    max_offset = inputFile.tell() - back_margin
    inputFile.seek(randint(0, max_offset), 0)
    inputfile.readline() # throw away an incomplete sentence

class SentencePairLoader():
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, inputFile, batch_size, tokenize, max_len, short_sampling_prob=0.1, pipeline=[]):
        super().__init__()
        self.f_pos = open(inputFile, "r", encoding='utf-8', errors='ignore')    # for a positive sample
        self.f_neg = open(inputFile, "r", encoding='utf-8', errors='ignore')    # for a negative (random) sample
        self.tokenize_fn = tokenize_fn  # tokenize function
        self.max_len = max_len          # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.pipeline = pipeline
        self.batch_size = batch_size

