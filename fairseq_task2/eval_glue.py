#!/usr/bin/env python3 -uimport sys
import sys
import importlib.util
import sys
from fairseq.models.roberta.model import RobertaModel

roberta = RobertaModel.from_pretrained(
    '/mnt/c/Users/yifei/OneDrive/桌面/CS8803EML/eml-hw2/roberta.base/',
    checkpoint_file='model.pt',
    data_name_or_path='RTE-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('glue_data/RTE/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
# with open('glue_data/RTE/dev.tsv') as fin:
#     fin.readline()
#     for index, line in enumerate(fin):
#         tokens = line.strip().split('\t')
#         sent1, sent2, target = tokens[1], tokens[2], tokens[3]
#         tokens = roberta.encode(sent1, sent2)
#         prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
#         prediction_label = label_fn(prediction)
#         ncorrect += int(prediction_label == target)
#         nsamples += 1
# print('| Accuracy: ', float(ncorrect)/float(nsamples))

# model_path = "/mnt/c/Users/yifei/OneDrive/桌面/CS8803EML/eml-hw2/quantized"
# torch.save(roberta.state_dict(), model_path)


