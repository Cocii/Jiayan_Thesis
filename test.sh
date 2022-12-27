#!/bin/bash
# sh test.sh
# python Dataset_complex_xf.py --train_session 1 --init_dataset_idx 1 --final_dataset_idx 1 &&
python Dataset_complex_xf.py --train_session 2 --init_dataset_idx 2 --final_dataset_idx 4 &&
python Dataset_complex_xf.py --train_session 2 --init_dataset_idx 5 --final_dataset_idx 7 &&
python Dataset_complex_xf.py --train_session 2 --init_dataset_idx 8 --final_dataset_idx 9