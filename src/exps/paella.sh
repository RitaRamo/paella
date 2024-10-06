args="""
    --experiments_dir paella
    --decoder_name facebook/xglm-2.9B
    --attention_size 1.75 
    --batch_size 16
    --gradient_steps 4
    --n_epochs 1
    --annotations_path data/dataset_coco_sample_all_35.json
    --multilingual
    --sampling_dataset

"""

python3 train.py $args 

