from functools import lru_cache
from .tokenizer_base import BaseTokenizer
from .tokenizer_re import TokenizerRegexp

import os
import sys
import subprocess
import tempfile
import itertools

class Tokenizer13a(BaseTokenizer):

    def signature(self):
        return '13a'

    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        """Tokenizes an input line using a relatively minimal tokenization
        that is however equivalent to mteval-v13a, used by WMT.

        :param line: a segment to tokenize
        :return: the tokenized line
        """

        # language-independent part:
        line = line.replace('<skipped>', '')
        line = line.replace('-\n', '')
        line = line.replace('\n', ' ')
        line = line.replace('.', '')

        if '&' in line:
            line = line.replace('&quot;', '"')
            line = line.replace('&amp;', '&')
            line = line.replace('&lt;', '<')
            line = line.replace('&gt;', '>')

        return self._post_tokenizer(f' {line} ')

    # def tokenize(self, captions_for_image):
    #     super
    # def tokenize(self, captions_for_image):
    #     cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR, \
    #             'edu.stanford.nlp.process.PTBTokenizer', \
    #             '-preserveLines', '-lowerCase']

    #     # ======================================================
    #     # prepare data for PTB Tokenizer
    #     # ======================================================
    #     final_tokenized_captions_for_image = {}
    #     image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]
    #     sentences = ([c['caption'].replace('\n', ' ') for k, v in captions_for_image.items() for c in v])


    #     print("captions_for_image",captions_for_image)
    #     print("sentences",sentences)
    #     # ======================================================
    #     # save sentences to temporary file
    #     # ======================================================
    #     # path_to_jar_dirname=os.path.dirname(os.path.abspath(__file__))
    #     # tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
    #     # tmp_file.write(sentences.encode())
    #     # tmp_file.close()

    #     # # ======================================================
    #     # # tokenize sentence
    #     # # ======================================================
    #     # cmd.append(os.path.basename(tmp_file.name))
    #     # p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
    #     #         stdout=subprocess.PIPE)
    #     # token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
    #     # token_lines = token_lines.decode()
    #     # lines = token_lines.split('\n')
    #     # # remove temp file
    #     # os.remove(tmp_file.name)

    #     # ======================================================
    #     # create dictionary for tokenized captions
    #     # ======================================================
    #     for k, line in zip(image_id, sentences):
    #         if not k in final_tokenized_captions_for_image:
    #             final_tokenized_captions_for_image[k] = []
    #         tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
    #                 if w not in PUNCTUATIONS])

            
    #         print("tokenized captiiiiiiiions", tokenized_caption)
    #         tokenized_caption = line.lower().rstrip()
    #         print("second", tokenized_caption)
    #         tokenized_caption = self.__call__(line.lower().rstrip())
    #         print("thirds", tokenized_caption)
            
    #         final_tokenized_captions_for_image[k].append(tokenized_caption)
        
    #     print("final_tokenized_captions_for_image",final_tokenized_captions_for_image)
    #     print(stop)
    #     return final_tokenized_captions_for_image


# if self.lowercase:
#             sent = sent.lower()
#         return self.tokenizer(sent.rstrip())