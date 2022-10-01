mkdir -p log
mkdir -p figures
python main.py --num_windows 1 \
              --seq_window_lengths 8 12 \
              --smi_window_lengths 4 8 \
              --batch_size 512 \
              --num_epoch 1 \
              --max_seq_len 1000 \
              --max_smi_len 100 \
              --dataset_path './data/davis/' \
              --problem_type 1 \
              --is_log 1