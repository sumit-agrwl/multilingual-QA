# python train_reader.py \
#         --train_data /projects/tir6/general/surajt/mQA/MIA-Shared-Task-2022/retrieval_output/mia_shared_train_dpr_retrieval_results_50.json \
#         --eval_data /projects/tir6/general/surajt/mQA/MIA-Shared-Task-2022/retrieval_output/mia_shared_xorqa_development_dpr_retrieval_results_50.json \
#         --model_size base \
#         --per_gpu_batch_size 1 \
#         --accumulation_steps 16 \
#         --n_context 50 \
#         --name dpr_retrieval_mia_50 \
#         --total_steps 600000 \
#         --eval_freq 50000 \
#         --save_freq 50000 \
#         --checkpoint_dir /projects/tir6/general/surajt/mQA/FiD/checkpoint \

# python train_reader.py \
#         --train_data /projects/tir6/general/surajt/mQA/MIA-Shared-Task-2022/retrieval_output/mia_shared_train_dpr_da_50.json \
#         --eval_data /projects/tir6/general/surajt/mQA/MIA-Shared-Task-2022/retrieval_output/mia_shared_xorqa_development_dpr_reranked_results_50.json \
#         --model_size base \
#         --per_gpu_batch_size 2 \
#         --accumulation_steps 8 \
#         --n_context 20 \
#         --name dpr_retrieval_mia_da_50_new \
#         --total_steps 400000 \
#         --eval_freq 50000 \
#         --save_freq 25000 \
#         --checkpoint_dir /projects/tir6/general/surajt/mQA/FiD/checkpoint \


# python train_reader.py \
#         --train_data /projects/tir6/general/surajt/mQA/MIA-Shared-Task-2022/retrieval_output/mia_shared_train_dpr_retrieval_results_50.json \
#         --eval_data /projects/tir6/general/surajt/mQA/MIA-Shared-Task-2022/retrieval_output/mia_shared_xorqa_development_dpr_reranked_results_50.json \
#         --model_size base \
#         --per_gpu_batch_size 2 \
#         --accumulation_steps 8 \
#         --n_context 20 \
#         --model_path /projects/tir6/general/surajt/mQA/FiD/checkpoint/dpr_retrieval_mia_mtl/checkpoint/best_dev \
#         --name dpr_retrieval_mia_after_mtl_20 \
#         --total_steps 300000 \
#         --eval_freq 50000 \
#         --save_freq 25000 \
#         --checkpoint_dir /projects/tir6/general/surajt/mQA/FiD/checkpoint \
        
python train_reader.py \
        --train_data /projects/tir6/general/surajt/mQA/MIA-Shared-Task-2022/retrieval_output/mia_shared_train_dpr_retrieval_results_50_gold.json \
        --eval_data /projects/tir6/general/surajt/mQA/MIA-Shared-Task-2022/retrieval_output/mia_shared_xorqa_development_dpr_reranked_results_200.json \
        --model_size base \
        --per_gpu_batch_size 1 \
        --accumulation_steps 16 \
        --n_context 50  \
        --name dpr_retrieval_mia_gold_50 \
        --total_steps 600000 \
        --eval_freq 50000 \
        --save_freq 50000 \
        --checkpoint_dir /projects/tir6/general/surajt/mQA/FiD/checkpoint \