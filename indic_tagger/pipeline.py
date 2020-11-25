import lstmcrf
from lstmcrf.utils import load_data_and_labels
from lstmcrf.wrapper import Sequence
import sys, os.path as path
import os
sys.path.append(path.dirname(path.abspath(__file__)))
import tagger.src.data_reader as data_reader
import tagger.src.generate_features as generate_features
import tagger.utils.writer as data_writer
import argparse
import logging
import pickle
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from tagger.src.algorithm.CRF import CRF
from polyglot_tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Scorer pipeline')
    #parser.add_argument("-p",'--pipeline_type', type=str, required=True,
                        #help='Pipeline Type (train, test, predict)')
    #parser.add_argument("-l", "--language", dest="language", type=str, metavar='<str>', required=True,
                         #help="Language of the dataset: te (telugu), hi (hindi), ta (tamil), ka (kannada), pu (pubjabi), mr (Marathi), be (Bengali), ur (Urdu), ml (Malayalam)")
    #parser.add_argument("-t", "--tag_type", dest="tag_type", type=str, metavar='<str>', required=True,
                         #help="Tag type: pos, chunk, parse, NER")
    #parser.add_argument("-m", "--model_type", dest="model_type", type=str, metavar='<str>', required=True,
                    #help="Model type (crf|hmm|cnn|lstm:) (default=crf)")
    #parser.add_argument("-e", "--encoding", dest="encoding", type=str, metavar='<str>', required=False,
                        #help="Encoding of the data (utf8, wx)",
                        #default="utf8")
    #parser.add_argument("-f", "--data_format", dest="data_format", type=str, metavar='<str>', required=True,
                       # help="Data format (ssf, tnt, txt)")
    #parser.add_argument("-i", "--input_file", dest="test_data", type=str, metavar='<str>', required=False,
                        #help="Test data path ex: data/test/te/test.txt")
    parser.add_argument("-s", "--sent_split", dest="sent_split", type=str, metavar='<str>', required=False,
                        help="Sentence Split ex: True or False",
                        default=True)
    #parser.add_argument("-o", "--output_file", dest="output_path", type=str, metavar='<str>',
                         #help="The path to the output file",
                         #default=path.join(path.dirname(path.abspath(__file__)), "outputs", "output_file"))
    return parser.parse_args()

def pipeline(input_path,output_path):
    curr_dir = path.dirname(path.abspath(__file__))
    #args = get_args()
    args.pipeline_type='predict'
    args.language='mr'
    args.tag_type='pos'
    args.model_type='crf'
    args.encoding='utf'
    args.data_format='txt'
    args.test_data=input_path
    args.output_path=output_path
    
    if True: 
        model_path = "%s/models/%s/%s.%s.%s.model" % (curr_dir, args.language, args.model_type, args.tag_type, args.encoding)
        if args.model_type == "lstm":
            if args.tag_type == "pos":
                model_path = "%s/models/%s/lstm/" % (curr_dir, args.language)
            elif args.tag_type == "chunk":
                model_path = "%s/models/%s/lstm/chunk/" % (curr_dir, args.language)
            elif args.tag_type == "ner":
                model_path = "%s/models/%s/lstm/ner/" % (curr_dir, args.language)
    if args.tag_type != "parse":
            if not os.path.exists(model_path):
                os.makedirs(model_path)

    if args.pipeline_type == "predict":

        test_data_path = "%s" % (args.test_data)
        test_sents = data_reader.load_data(args.data_format, test_data_path, args.language, tokenize_text=True, split_sent=args.sent_split)
        if args.tag_type == "parse":
            #Pos tagging
            X_test = [ generate_features.sent2features(s, "pos", args.model_type) for s in test_sents ]

            tag_model_path = "%s/models/%s/%s.%s.%s.model" % (curr_dir, args.language, args.model_type, "pos", args.encoding)
            chunk_model_path = "%s/models/%s/%s.%s.%s.model" % (curr_dir, args.language, args.model_type, "chunk", args.encoding)

            if args.model_type == "crf":
                tagger = CRF(tag_model_path)
                tagger.load_model()
                y_pos = tagger.predict(X_test)

                test_sents_pos = generate_features.append_tags(test_sents, "pos", y_pos)
                X_test = [ generate_features.sent2features(s, "chunk", args.model_type) for s in test_sents_pos ]

                chunker = CRF(chunk_model_path)
                chunker.load_model()
                y_chunk = chunker.predict(X_test)

                test_fname = path.basename(test_data_path)
                output_file = "%s/%s.parse" % (output_dir, test_fname)
                data_writer.write_anno_to_file(output_file, test_sents_pos, y_chunk, "chunk")
                logger.info("Output in: %s" % output_file)
                data_writer.write_to_screen(output_file)
        else:
            X_test = [ generate_features.sent2features(s, args.tag_type, args.model_type) for s in test_sents ]

            if args.model_type == "crf":
                tagger = CRF(model_path)
                tagger.load_model()
                y_pred = tagger.predict(X_test)
                data_writer.write_anno_to_file(args.output_path, test_sents, y_pred, args.tag_type)
                data_writer.write_to_screen(args.output_path)
                logger.info("Output in: %s" % args.output_path)

            if args.model_type == "lstm":
                model = Sequence().load(model_path+"/weights.h5", model_path+"/params.json", model_path+"/preprocessor.json")
                f = open(args.test_data, "r")
                sent = f.read()
                tok = Tokenizer(lang=args.language, split_sen=True)
                tokenized_sents = tok.tokenize(sent)
                for tokens in tokenized_sents:
                    for token in tokens:
                          sent = sent + " " + token
                    sent = sent.strip()
                    print(model.analyze(sent))

if __name__ == '__main__':
    pipeline()
