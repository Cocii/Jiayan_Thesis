#!/bin/bash
# sh test.sh

### downfactor = 2  training
python Dataset_complex_xf_train.py --train_session 1 --init_dataset_idx 1 --final_dataset_idx 1 &&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 2 --final_dataset_idx 3 &&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 4 --final_dataset_idx 5 &&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 6 --final_dataset_idx 7 &&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 8 --final_dataset_idx 9 &&

### downfactor = 4  training
python Dataset_complex_xf_train.py --train_session 1 --init_dataset_idx 1 --final_dataset_idx 1 --down_factor 4 --outdir '../ModelCheckpoint/super_res_imag_xf_4.h5' --outdir_trainhistory '../ModelCheckpoint/th_imag_xf_4' --outdir_metrics '../Metrics/Metrics_behaviour_imag_xf_4' --outdir_plots '../Plots/Plot_imag_xf_4'&&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 2 --final_dataset_idx 3 --down_factor 4 --outdir '../ModelCheckpoint/super_res_imag_xf_4.h5' --outdir_trainhistory '../ModelCheckpoint/th_imag_xf_4' --outdir_metrics '../Metrics/Metrics_behaviour_imag_xf_4' --outdir_plots '../Plots/Plot_imag_xf_4'&&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 4 --final_dataset_idx 5 --down_factor 4 --outdir '../ModelCheckpoint/super_res_imag_xf_4.h5' --outdir_trainhistory '../ModelCheckpoint/th_imag_xf_4' --outdir_metrics '../Metrics/Metrics_behaviour_imag_xf_4' --outdir_plots '../Plots/Plot_imag_xf_4'&&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 6 --final_dataset_idx 7 --down_factor 4 --outdir '../ModelCheckpoint/super_res_imag_xf_4.h5' --outdir_trainhistory '../ModelCheckpoint/th_imag_xf_4' --outdir_metrics '../Metrics/Metrics_behaviour_imag_xf_4' --outdir_plots '../Plots/Plot_imag_xf_4'&&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 8 --final_dataset_idx 9 --down_factor 4 --outdir '../ModelCheckpoint/super_res_imag_xf_4.h5' --outdir_trainhistory '../ModelCheckpoint/th_imag_xf_4' --outdir_metrics '../Metrics/Metrics_behaviour_imag_xf_4' --outdir_plots '../Plots/Plot_imag_xf_4'&&

### downfactor = 8  training
python Dataset_complex_xf_train.py --train_session 1 --init_dataset_idx 1 --final_dataset_idx 1 --down_factor 8 --outdir '../ModelCheckpoint/super_res_imag_xf_8.h5' --outdir_trainhistory '../ModelCheckpoint/th_imag_xf_8' --outdir_metrics '../Metrics/Metrics_behaviour_imag_xf_8' --outdir_plots '../Plots/Plot_imag_xf_8'&&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 2 --final_dataset_idx 3 --down_factor 8 --outdir '../ModelCheckpoint/super_res_imag_xf_8.h5' --outdir_trainhistory '../ModelCheckpoint/th_imag_xf_8' --outdir_metrics '../Metrics/Metrics_behaviour_imag_xf_8' --outdir_plots '../Plots/Plot_imag_xf_8'&&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 4 --final_dataset_idx 5 --down_factor 8 --outdir '../ModelCheckpoint/super_res_imag_xf_8.h5' --outdir_trainhistory '../ModelCheckpoint/th_imag_xf_8' --outdir_metrics '../Metrics/Metrics_behaviour_imag_xf_8' --outdir_plots '../Plots/Plot_imag_xf_8'&&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 6 --final_dataset_idx 7 --down_factor 8 --outdir '../ModelCheckpoint/super_res_imag_xf_8.h5' --outdir_trainhistory '../ModelCheckpoint/th_imag_xf_8' --outdir_metrics '../Metrics/Metrics_behaviour_imag_xf_8' --outdir_plots '../Plots/Plot_imag_xf_8'&&
python Dataset_complex_xf_train.py --train_session 2 --init_dataset_idx 8 --final_dataset_idx 9 --down_factor 8 --outdir '../ModelCheckpoint/super_res_imag_xf_8.h5' --outdir_trainhistory '../ModelCheckpoint/th_imag_xf_8' --outdir_metrics '../Metrics/Metrics_behaviour_imag_xf_8' --outdir_plots '../Plots/Plot_imag_xf_8'