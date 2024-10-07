from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import sys
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and pathes
dataDir='coco-caption'
#dataType = sys.argv[1]
algName = 'fakecap'
annFile=sys.argv[1]#'%s/annotations/captions_%sKarpathy.json'%(dataDir,dataType)
subtypes=['results', 'evalImgs', 'eval']

resFile = sys.argv[2]
language = sys.argv[3]

print("ola")
print("annFile",annFile)
print("language",language)
print("resFile",resFile)
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate(language)


each_image_score = {}
individual_scores = [eva for eva in cocoEval.evalImgs]
count_exception=0
for i in range(len(individual_scores)):
    coco_id = individual_scores[i]["image_id"]
    each_image_score[coco_id] = individual_scores[i]



outfile = resFile.replace('preds', 'res')
outfile = outfile.replace('json', 'txt')

with open(outfile, 'w') as outfile:
  for metric, score in cocoEval.eval.items():
    outfile.write( '%s: %.2f\n'%(metric, score*100) )

outfile = resFile.replace('preds', 'res_ind.json')

with open(outfile, 'w') as outfile:
    json.dump(each_image_score, outfile, ensure_ascii=False)
