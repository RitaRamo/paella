__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .tokenizers.tokenizer_13a import Tokenizer13a
from .tokenizers.tokenizer_zh import TokenizerZh
from .tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
from .tokenizers.tokenizer_ko_mecab import TokenizerKoMecab
from .tokenizers.tokenizer_intl import TokenizerV14International
from .tokenizers.tokenizer_thai import TokenizerThai
from .tokenizers.tokenizer_bn import TokenizerBengali

from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice

_TOKENIZERS = {
    'zh': TokenizerZh(),
    '13a': Tokenizer13a(),
    'en': PTBTokenizer(),
    'ja': TokenizerJaMecab(),
    'ko': TokenizerKoMecab(),
    'th': TokenizerThai(),
    'bn': TokenizerBengali(),
    #'te': TokenizerV14International()

}


class COCOEvalCap:


    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self,language):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        #tokenizer = PTBTokenizer()


        tokenizer = TokenizerZh()

        # if language =="zh" or language =="ja":
        if language =="zh" or language =="en" or language=="ja" or language =="ko"  or language=="th" or language=="bn":
            tokenizer = _TOKENIZERS[language]
        else:
            tokenizer = _TOKENIZERS["13a"]
        
        print("language and tokenizer", language, tokenizer)
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)


        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            #(Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
