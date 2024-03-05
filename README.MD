# Task 2 
## modifications:
### Quantization
1. implemented QLinear with reference from https://github.com/GATECH-EIC/CPT/blob/master/cpt_cifar/modules/quantize.py QConv class. Specifically, code located in fairseq\fairseq\modules\quantize.py starting from line 255.

2. replacing all nn.linear layer from multihead attention(fairseq/fairseq/modules/multihead_attention.py) with QLinear, specifically the k_proj, v_proj, q_proj, and out_proj(this makes it automatically call quanlize during forward pass in training). Then, do a separate quantization by calling quantize_input() from CPT on k,v,q, and out vector before vector multiplication.

3. replace the declaration of nn.Linear for fc1 and fc2 with QLinear in the feed forwward network(fairseq\fairseq\modules\transformer_layer.py)

4. I would manually modify the paramter num_bits from attention/feedforward network file when running different tasks. This enables me to run different quantization on different layer for task 2 quetion 2's result(for example when running 4 bits for feedforward and 16 bits for multihead attention).

### Freeze backbone layers
1. in train.py main(), before loading model into trainer, loop through each layer to modify the require_grad parameter to False for all layer whose name does not match 'classification_head'. Implemented in fairseq_task2\fairseq_cli\train.py, line 148.

## command to run(running RTE for example)
CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name rte 'task.data="/mnt/c/Users/yifei/OneDrive/桌面/CS8803EML/eml-hw2/fairseq_task2/RTE-bin"' 'checkpoint.restore_file="/mnt/c/Users/yifei/OneDrive/桌面/CS8803EML/eml-hw2/roberta.base/model.pt"' 

# Task 3.
## modification
1. I added the function cyclic_adjust_precision() form CPT to the trainer object. In the function, I deleted all calculation related to gradient, then I modified the function such that it does not pass in the same parameters, instead it saves it as class variable, making it accessable across functions in the class. (starting from fairseq\fairseq\trainer.py line 802)

2. I called cyclic_adjust precision in train_step, and calculated cyclic period before calling it(starting from fairseq\fairseq\trainer.py line 825).

3. I kept the Qlinear and quantization from task 2, however, I passed the num_bits as input parameter from trainer's train_step all the way to the base model in order to make it automatically adjust precision. some affected files include(with only parameters modified in the forward function):
fairseq\fairseq\models\roberta\model.py
fairseq\fairseq\models\transformer\transformer_encoder.py
fairseq\fairseq\tasks\fairseq_task.py

## command to run(running RTE for example)
CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name rte 'task.data="/mnt/c/Users/yifei/OneDrive/桌面/CS8803EML/eml-hw2/fairseq_task2/RTE-bin"' 'checkpoint.restore_file="/mnt/c/Users/yifei/OneDrive/桌面/CS8803EML/eml-hw2/roberta.base/model.pt"' 
