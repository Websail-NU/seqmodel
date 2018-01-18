#!/bin/bash

# PTB

# python main_lm.py train data/ptb explm/stat/ptb-pre --gpu --log_level debug --load_model_opt explm/stat/ptb-config/model_opt.json --load_train_opt explm/stat/ptb-config/train_opt.json --train:max_epoch 5 --batch_size 20 --seq_len 35
# cp -r explm/stat/ptb-pre explm/stat/ptb-v
# mkdir explm/stat/ptb-v/decode
# python main_lm.py train data/ptb explm/stat/ptb-v --gpu --log_level debug --load_model_opt explm/stat/ptb-config/model_opt.json --load_train_opt explm/stat/ptb-config/train_opt.json --train:max_epoch 20 --batch_size 20 --seq_len 35
# python main_lm.py decode data/ptb tmp --load_checkpoint explm/stat/ptb-v/checkpoint/best --gpu --log_level debug --seq_len 35 --batch_size 64 --load_model_opt explm/stat/ptb-config/model_opt.json --out:decode --decode:outpath explm/stat/ptb-v/decode/final.txt



# DATA='ptb'
# V='2'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 1 --gns:min_p_count 1

# DATA='ptb'
# V='3'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 3 --gns:ngram_min_order 3 --gns:use_rep \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 1 --gns:min_p_count 1

# DATA='ptb'
# V='4'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 4 --gns:ngram_min_order 4 --gns:use_rep \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 1 --gns:min_p_count 1


# DATA='ptb'
# V='6'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 2 --gns:min_p_count 2 --gns:alpha 0.25

# DATA='ptb'
# V='7'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 200 --gns:percent_new_tokens 0.15 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 5 --gns:min_p_count 5 --gns:alpha 0.5

# DATA='ptb'
# V='8'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 200 --gns:percent_new_tokens 0.15 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 5 --gns:min_p_count 5 --gns:alpha 1.0 --gns:clip_ratio 2.0

# DATA='ptb'
# V='9'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:min_p0_count 2 --gns:min_p_count 2 --gns:alpha 0.5 --gns:use_model_prob \
# --gns:precompute_after_steps 400 --gns:percent_new_tokens -1 --gns:clip_ratio 2.0


# DATA='ptb'
# V='11'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 200 --gns:percent_new_tokens 0.15 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 2 --gns:min_p_count 2 --gns:alpha 0.5 --gns:clip_ratio 2.0 \
# --gns:num_constraints_per_token 2000

# wikitext-2

# python main_lm.py train data/wikitext-2 explm/stat/wikitext-2-pre --gpu --log_level debug --seq_len 35 --batch_size 64 --load_model_opt explm/stat/wikitext-2-config/model_opt.json --load_train_opt explm/stat/wikitext-2-config/train_opt.json --train:max_epoch 5
# cp -r explm/stat/wikitext-2-pre explm/stat/wikitext-2-v
# mkdir explm/stat/wikitext-2-v/decode
# python main_lm.py train data/wikitext-2 explm/stat/wikitext-2-v --gpu --log_level debug --seq_len 35 --batch_size 64 --load_model_opt explm/stat/wikitext-2-config/model_opt.json --load_train_opt explm/stat/wikitext-2-config/train_opt.json
# python main_lm.py decode data/wikitext-2 tmp --load_checkpoint explm/stat/wikitext-2-v/checkpoint/best --gpu --log_level debug --seq_len 35 --batch_size 64 --load_model_opt explm/stat/wikitext-2-config/model_opt.json --out:decode --decode:outpath explm/stat/wikitext-2-v/decode/final.txt

# DATA='wikitext-2'
# V='1'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 64 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt --gns:remove_sen \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 2200000 \
# --gns:min_p0_count 10 --gns:min_p_count 10

# DATA='wikitext-2'
# V='2'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 64 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt --gns:remove_sen \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 2200000 \
# --gns:min_p0_count 1 --gns:min_p_count 1


# WordNet

# python main_word2def.py train data/wn_lemma_senses/ explm/stat/wn-lemma-pre --gpu --log_level debug --batch_size 64 --load_model_opt explm/stat/dm-config/model_opt.json --load_train_opt explm/stat/dm-config/train_opt.json --train:max_epoch 5
# cp -r explm/stat/wn-lemma-pre explm/stat/wn-lemma-v
# mkdir explm/stat/wn-lemma-v/decode
# python main_word2def.py train data/wn_lemma_senses explm/stat/wn-lemma-v --gpu --log_level debug --batch_size 64 --load_model_opt explm/stat/dm-config/model_opt.json --load_train_opt explm/stat/dm-config/train_opt.json
# python main_word2def.py decode data/wn_lemma_senses tmp --load_checkpoint explm/stat/wn-lemma-v/checkpoint/best --gpu --log_level debug --batch_size 64 --load_model_opt explm/stat/dm-config/model_opt.json --decode:outpath explm/stat/wn-lemma-v/decode/final.txt --eval_file train.txt

# RUN='wn-lemma'
# V='1'
# source prep_exp_dir.sh $RUN m$V
# python main_global_stat_dm.py train data/wn_lemma_senses explm/stat/$RUN-m$V --gpu \
# --log_level debug --batch_size 64 \
# --load_model_opt explm/stat/dm-config/model_opt.json \
# --load_train_opt explm/stat/dm-config/train_opt.json \
# --gns:ref_text_path data/wn_lemma_senses/train_defs.txt --gns:dec_batch_size 64 \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 1000000 \
# --gns:min_p0_count 1 --gns:min_p_count 1 --gns:dec_batch_size 256

# RUN='wn-lemma'
# V='2'
# source prep_exp_dir.sh $RUN m$V
# python main_global_stat_dm.py train data/wn_lemma_senses explm/stat/$RUN-m$V --gpu \
# --log_level debug --batch_size 64 \
# --load_model_opt explm/stat/dm-config/model_opt.json \
# --load_train_opt explm/stat/dm-config/train_opt.json \
# --gns:ref_text_path data/wn_lemma_senses/train_defs.txt --gns:dec_batch_size 64 \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 1000000 \
# --gns:min_p0_count 1 --gns:min_p_count 1 --gns:dec_batch_size 256 --gns:dec_temperature 0.1 \
# --gns:clip_ratio 5.0 --gns:uniq_dec_text


# RUN='wn-lemma'
# V='3'
# source prep_exp_dir.sh $RUN m$V
# python main_global_stat_dm.py train data/wn_lemma_senses explm/stat/$RUN-m$V --gpu \
# --log_level debug --batch_size 64 \
# --load_model_opt explm/stat/dm-config/model_opt.json \
# --load_train_opt explm/stat/dm-config/train_opt.json \
# --gns:ref_text_path data/wn_lemma_senses/train_defs.txt --gns:dec_batch_size 64 \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 1000000 \
# --gns:min_p0_count 1 --gns:min_p_count 1 --gns:dec_batch_size 256 --gns:clip_ratio 5.0


# RUN='wn-lemma'
# V='4'
# source prep_exp_dir.sh $RUN m$V
# python main_global_stat_dm.py train data/wn_lemma_senses explm/stat/$RUN-m$V --gpu \
# --log_level debug --batch_size 64 \
# --load_model_opt explm/stat/dm-config/model_opt.json \
# --load_train_opt explm/stat/dm-config/train_opt.json \
# --gns:ref_text_path data/wn_lemma_senses/train_defs.txt --gns:dec_batch_size 64 \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 1000000 \
# --gns:min_p0_count 2 --gns:min_p_count 2 --gns:dec_batch_size 256 --gns:clip_ratio 2.0 \
# --gns:alpha 0.5


# RUN='wn-lemma'
# V='15'
# source prep_exp_dir.sh $RUN m$V
# python main_global_stat_dm.py train data/wn_lemma_senses explm/stat/$RUN-m$V --gpu \
# --log_level debug --batch_size 64 \
# --load_model_opt explm/stat/dm-config/model_opt.json \
# --load_train_opt explm/stat/dm-config/train_opt.json \
# --gns:ref_text_path data/wn_lemma_senses/train_defs.txt --gns:dec_batch_size 256 \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_repm \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 1000000 \
# --gns:min_p0_count 2 --gns:min_p_count 1 --gns:clip_ratio 5.0 \
# --gns:alpha 1.0

# RUN='wn-lemma'
# V='17'
# source prep_exp_dir.sh $RUN m$V
# python main_global_stat_dm.py train data/wn_lemma_senses explm/stat/$RUN-m$V --gpu \
# --log_level debug --batch_size 64 \
# --load_model_opt explm/stat/dm-config/model_opt.json \
# --load_train_opt explm/stat/dm-config/train_opt.json \
# --gns:ref_text_path data/wn_lemma_senses/train_defs.txt --gns:dec_batch_size 256 \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 1000000 \
# --gns:min_p0_count 2 --gns:min_p_count 1 --gns:clip_ratio 5.0 \
# --gns:alpha 1.0 --gns:add_repk_kld --gns:avg_C_size 25 --gns:avg_unigram_size 25 \
# --gns:average_repk_size 25



# RUN='wn-lemma'
# V='9'
# source prep_exp_dir.sh $RUN m$V
# python main_global_stat_dm.py train data/wn_lemma_senses explm/stat/$RUN-m$V --gpu \
# --log_level debug --batch_size 64 \
# --load_model_opt explm/stat/dm-config/model_opt.json \
# --load_train_opt explm/stat/dm-config/train_opt.json \
# --gns:ref_text_path data/wn_lemma_senses/train_defs.txt --gns:dec_batch_size 256 \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 1000000 \
# --gns:min_p0_count 2 --gns:min_p_count 1 --gns:clip_ratio 5.0 \
# --gns:alpha 1.0

# RUN='wn-lemma'
# V='10'
# source prep_exp_dir.sh $RUN m$V
# python main_global_stat_dm.py train data/wn_lemma_senses explm/stat/$RUN-m$V --gpu \
# --log_level debug --batch_size 64 \
# --load_model_opt explm/stat/dm-config/model_opt.json \
# --load_train_opt explm/stat/dm-config/train_opt.json \
# --gns:ref_text_path data/wn_lemma_senses/train_defs.txt --gns:dec_batch_size 256 \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 1000000 \
# --gns:min_p0_count 2 --gns:min_p_count 1 --gns:clip_ratio 5.0 \
# --gns:alpha 0.5


# DATA='ptb'
# V='22'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 200 --gns:percent_new_tokens 0.15 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 2 --gns:min_p_count 2 --gns:alpha 0.5 --gns:add_unigram_kld \
# --gns:avg_C_size 25 --gns:avg_unigram_size 25  --gns:clip_ratio 2.0 --gns:use_model_prob \
# --gns:precompute_after_steps 400 --gns:percent_new_tokens -1


# DATA='ptb'
# V='23'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 200 --gns:percent_new_tokens 0.15 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 2 --gns:min_p_count 1 --gns:clip_ratio 2.0 \
# --gns:alpha 1.0 --gns:add_repk_kld --gns:avg_C_size 25 --gns:avg_unigram_size 25 \
# --gns:average_repk_size 25 --gns:use_model_prob \
# --gns:precompute_after_steps 400 --gns:percent_new_tokens -1


# RUN='wn-lemma'
# V='20'
# source prep_exp_dir.sh $RUN m$V
# python main_global_stat_dm.py train data/wn_lemma_senses explm/stat/$RUN-m$V --gpu \
# --log_level debug --batch_size 64 \
# --load_model_opt explm/stat/dm-config/model_opt.json \
# --load_train_opt explm/stat/dm-config/train_opt.json \
# --gns:ref_text_path data/wn_lemma_senses/train_defs.txt --gns:dec_batch_size 256 \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 1000000 \
# --gns:min_p0_count 2 --gns:min_p_count 2 --gns:clip_ratio 5.0 \
# --gns:alpha 0.5 --gns:add_repk_kld --gns:avg_C_size 25 --gns:avg_unigram_size 25 \
# --gns:average_repk_size 25 --gns:add_unigram_kld


# RUN='wn-lemma'
# V='19'
# source prep_exp_dir.sh $RUN m$V
# python main_global_stat_dm.py train data/wn_lemma_senses explm/stat/$RUN-m$V --gpu \
# --log_level debug --batch_size 64 \
# --load_model_opt explm/stat/dm-config/model_opt.json \
# --load_train_opt explm/stat/dm-config/train_opt.json \
# --gns:ref_text_path data/wn_lemma_senses/train_defs.txt --gns:dec_batch_size 256 \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 1000000 \
# --gns:min_p0_count 2 --gns:min_p_count 1 --gns:clip_ratio 5.0 \
# --gns:alpha 1.0 --gns:add_repk_kld --gns:avg_C_size 25 --gns:avg_unigram_size 25 \
# --gns:average_repk_size 25 --gns:dec_temperature 0.1 --gns:loss_temperature 1.0 --gns:uniq_dec_text



# DATA='ptb'
# V='24'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_repm \
# --gns:precompute_after_steps 200 --gns:percent_new_tokens 0.15 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 2 --gns:min_p_count 1 --gns:clip_ratio 5.0 \
# --gns:alpha 1.0 --gns:avg_C_size 25 --gns:avg_unigram_size 25 \
# --gns:average_repk_size 25


# RUN='wn-lemma'
# V='22'
# source prep_exp_dir.sh $RUN m$V
# python main_global_stat_dm.py train data/wn_lemma_senses explm/stat/$RUN-m$V --gpu \
# --log_level debug --batch_size 64 \
# --load_model_opt explm/stat/dm-config/model_opt.json \
# --load_train_opt explm/stat/dm-config/train_opt.json \
# --gns:ref_text_path data/wn_lemma_senses/train_defs.txt --gns:dec_batch_size 256 \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.10 --gns:dec_total_tokens 1000000 \
# --gns:min_p0_count 2 --gns:min_p_count 2 --gns:clip_ratio 5.0 \
# --gns:alpha 0.5 --gns:add_repk_kld --gns:avg_C_size 25 --gns:avg_unigram_size 25 \
# --gns:average_repk_size 25

# DATA='ptb'
# V='25'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 200 --gns:percent_new_tokens 0.15 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 2 --gns:min_p_count 2 --gns:alpha 0.5  \
# --gns:avg_C_size 25 --gns:avg_unigram_size 25  --gns:clip_ratio 5.0

# DATA='ptb'
# V='26'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 200 --gns:percent_new_tokens 0.15 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 2 --gns:min_p_count 2 --gns:alpha 0.5 \
# --gns:avg_C_size 25 --gns:avg_unigram_size 25  --gns:clip_ratio 2.0 --gns:use_model_prob \
# --gns:precompute_after_steps 400 --gns:percent_new_tokens -1


# DATA='ptb'
# V='29'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 200 --gns:percent_new_tokens 0.15 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 2 --gns:min_p_count 1 --gns:clip_ratio 5.0 \
# --gns:alpha 1.0 --gns:avg_C_size 25 --gns:avg_unigram_size 25 \
# --gns:average_repk_size 25 --gns:add_repk_kld

# DATA='ptb'
# V='28'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 200 --gns:percent_new_tokens 0.15 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 2 --gns:min_p_count 1 --gns:clip_ratio 2.0 \
# --gns:alpha 1.0 --gns:avg_C_size 25 --gns:avg_unigram_size 25 \
# --gns:average_repk_size 25 --gns:use_model_prob \
# --gns:precompute_after_steps 400 --gns:percent_new_tokens -1

# DATA='wikitext-2'
# V='7'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 64 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/wikitext-2/train.txt --gns:remove_sen --gns:remove_unk \
# --gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
# --gns:precompute_after_steps 100 --gns:percent_new_tokens 0.1 --gns:dec_total_tokens 2200000 \
# --gns:min_p0_count 5 --gns:min_p_count 5 --gns:num_constraints_per_token 5000 \
# --gns:alpha 0.5 --gns:clip_ratio 2.0 --gns:num_processes 12 --gns:add_unigram_kld  \
# --gns:avg_C_size 25 --gns:avg_unigram_size 25


# DATA='ptb'
# V='31'
# source prep_exp_dir.sh $DATA m$V
# python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
# --log_level debug --batch_size 20 --seq_len 35 \
# --load_model_opt explm/stat/$DATA-config/model_opt.json \
# --load_train_opt explm/stat/$DATA-config/train_opt.json \
# --gns:ref_text_path data/$DATA/train.txt \
# --gns:ngram_max_order 4 --gns:ngram_min_order 2 --gns:use_rep \
# --gns:precompute_after_steps 200 --gns:percent_new_tokens 0.15 --gns:dec_total_tokens 929589 \
# --gns:min_p0_count 2 --gns:min_p_count 1 --gns:clip_ratio 5.0 \
# --gns:alpha 1.0 --gns:avg_C_size 20 --gns:avg_unigram_size 1 \
# --gns:average_repk_size 20 --gns:text_history_size 1 --gns:add_repk_kld



DATA='ptb'
V='42'
source prep_exp_dir.sh $DATA m$V
python main_global_stat_lm.py train data/$DATA explm/stat/$DATA-m$V --gpu \
--log_level debug --batch_size 20 --seq_len 35 \
--load_model_opt explm/stat/$DATA-config/model_opt.json \
--load_train_opt explm/stat/$DATA-config/train_opt.json \
--gns:ref_text_path data/$DATA/train_fake.txt \
--gns:ngram_max_order 2 --gns:ngram_min_order 2 --gns:use_lm \
--gns:precompute_after_steps 100 --gns:percent_new_tokens 0.1 --gns:dec_total_tokens 929589 \
--gns:min_p0_count 2 --gns:min_p_count 2 --gns:alpha 1.0  \
--gns:avg_C_size 100 --gns:avg_unigram_size 100  --gns:clip_ratio 5.0 \
--gns:text_history_size 100 --gns:add_unigram_kld  --train:max_epoch 15
