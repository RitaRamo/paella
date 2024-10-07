LanguageArray=('ar' 'bn' 'cs' 'da' 'de' 'el' 'en' 'es' 'fa' 'fi' 'fil' 'fr' 'iw' 'hi' 'hr' 'hu' 'id' 'it' 'ja' 'ko' 'mi' 'nl' 'no' 'pl' 'pt' 'ro' 'ru' 'sv' 'sw' 'te' 'th' 'tr' 'uk' 'vi' 'zh')
CheckpointArray=( "26556" )

for checkpoint in ${CheckpointArray[*]}; do
    for LANGUAGE in ${LanguageArray[*]}; do

        args="""
            
            --model_path exps_paella/rag_1.75M_facebook/xglm-2.9B \
            --decoder_name facebook/xglm-2.9B
            --checkpoint_path checkpoint-${checkpoint}
            --template_path src/templates/template_${LANGUAGE}.txt
            --captions_path data/retrieved_caps/xm3600/retrieved_ids_resnet50x64.${LANGUAGE}.json
            --features_path features_xm3600/val.hdf5
            --annotations_path data/xm3600/annotations/captions_val2014_new.json
            --dataset xm3600
        """

        python3 infer.py $args
    done
done
