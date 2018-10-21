import json, logging, os, data
import torch
import torch.nn as nn
import boto3

from model import EncoderRNN, Attn, LuongAttnDecoderRNN
from evaluate import GreedySearchDecoder, predict_answer
from sagemaker import get_execution_role

role = get_execution_role()
bucketName = 'faq-chatbot'

JSON_CONTENT_TYPE = 'application/json'
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info('Current device: {}'.format(device))

def model_fn(model_dir):
    logger.info('Loading the model.')
    model_info = {}

    with open(os.path.join(model_dir, 'model_info.pth'), 'rb') as f:
        # If loading a model trained on GPU to CPU
        if torch.cuda.device_count() < 1:
            checkpoint = torch.load(f, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(f)
        
        #have to save these hyper parameters
        hidden_size = model_info['hidden_size']
        encoder_n_layers = model_info['encoder_n_layers']
        decoder_n_layers = model_info['decoder_n_layers']
        dropout = model_info['dropout']
        attn_model = model_info['attn_model']
        voc = model_info['voc']
        
        # Initialize word embeddings
        embedding = nn.Embedding(voc.num_words, hidden_size)
        embedding.load_state_dict(checkpoint['embedding'])

        # Initialize encoder & decoder models
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
        
        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()
        
        searcher = GreedySearchDecoder(encoder, decoder, device)
        
        return {'searcher': searcher, 'voc': voc}


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info('Deserializing the input data.')

    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        if len(input_data['question']) < 3:
            raise Exception('\'question\' has to be larger than 3 char')
        return input_data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    logger.info('Generating answer based on input question.')
    with torch.no_grad():  # no tracking history
        return ''.join(predict_answer(model['searcher'], model['voc'], input_data, device))
        
    
    
    



