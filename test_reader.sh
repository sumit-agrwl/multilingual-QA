python train_reader.py \
        --train_data /projects/tir6/general/surajt/mQA/MIA-Shared-Task-2022/retrieval_output/mia_shared_train_dpr_retrieval_results_50.json \
        --eval_data /projects/tir6/general/surajt/mQA/MIA-Shared-Task-2022/retrieval_output/mia_shared_mkqa_development_dpr_retrieval_results_200_tr.json \
        --model_size base \
        --per_gpu_batch_size 1 \
        --accumulation_steps 16 \
        --n_context 50 \
        --name dpr_retrieval_mia_50 \
        --model_path /projects/tir6/general/surajt/mQA/FiD/checkpoint/dpr_retrieval_mia_50/checkpoint/best_dev/ \
        --checkpoint_dir /projects/tir6/general/surajt/mQA/FiD/checkpoint \