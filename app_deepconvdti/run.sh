mkdir -p log
python main.py ./data/training_dataset/training_dti.csv \
            ./data/training_dataset/training_compound.csv \
            ./data/training_dataset/training_protein.csv \
            --validation -n validation_dataset -i ./data/validation_dataset/validation_dti.csv \
            -d ./data/validation_dataset/validation_compound.csv -t ./data/validation_dataset/validation_protein.csv \
            -W -c 512 128 -w 10 15 20 25 30 -p 128 -f 128 -r 0.0001 -n 30 -v Convolution -l 2500 -V morgan_fp_r2 -L 2048 -D 0 -a elu -F 128 -b 32 -y 0.0001\
            -o ./validation_output.csv -m ./model.model -e 1