
args="""
    --experiments_dir paella_core
    --decoder_name facebook/xglm-2.9B
    --attention_size 1.75 
    --batch_size 16
    --gradient_steps 4
    --n_epochs 3
    --annotations_path data/dataset_coco_sample_core.json
    --multilingual
"""

python3 train.py $args 