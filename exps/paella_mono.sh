args="""
    --experiments_dir paella_mono_en
    --decoder_name facebook/xglm-2.9B
    --attention_size 1.75 
    --batch_size 16
    --gradient_steps 4
    --template_path src/templates/template.txt
    --captions_path data/retrieved_caps/retrieved_ids_resnet50x64.en.json
    --n_epochs 3
"""

python3 train.py $args 


# args="""
#     --experiments_dir pealla_mono_da
#     --decoder_name facebook/xglm-2.9B
#     --attention_size 1.75 
#     --batch_size 16
#     --gradient_steps 4
#     --template_path src/templates/template_da.txt
#     --n_epochs 3
#     --translation
#     --captions_path data/retrieved_caps/retrieved_ids_resnet50x64.da.json
#     --annotations_path data/dataset_coco_mt_da.json
# """

# python3 train.py $args 
