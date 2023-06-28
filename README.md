# ACP-PDAFF:Pretrained model and dual-channel attentional feature fusion for anticancer peptide prediction



## How to use it

### Pretrain code

You should download prot_bert_bfd （https://huggingface.co/Rostlab/prot_bert_bfd） into prot_bert_bfd folder before you use, then run code in pretrain.py and save the pretrain embedding.

### Train-test code

The main program in the train folder ACP_PDAFF_main.py file. You could change the load_config function to achieve custom training and testing, such as modifying datasets, setting hyperparameters and so on. File ACP_PDAFF_main.py has detail notes.



