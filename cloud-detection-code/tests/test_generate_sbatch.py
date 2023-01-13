"""
Test generate_sbatch.py
"""
import os

def test_resume_and_from_start():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 7 7 7 3 -l mse mse focal focal mse focal --lr 0.001 0.001 0.001 0.001 0.001 0.001 -m dense dense dense non non dense -e 1000 1000 1000 1000 1000 1000 -s 885 780 790 645 640 0 -d 2022-08-05 2022-08-05 2022-08-05 2022-08-05 2022-08-05 2022-08-05"
    os.system(test_gen)

    #Compare generated to expected sbatch files
    fnames = ['3x3_dense_focal_0_001_model_checkpoints_5.sh', '7x7_nondense_mse_0_001_model_checkpoints_4.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_learning_rate_schedule():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_schedule'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l mse focal --lr 0.001 -m dense non -e 1000 -s 885 0 -d 2022-08-05 --lr_epochs 1000 --lr_epochs 300 500 --lr_rates 0.0001 --lr_rates 0.0001 0.00005 --load-no-schedule" 
    os.system(test_gen)

    fnames = ['3x3_dense_mse_0_001_lr_1000_0_0001_model_checkpoints_0.sh', '7x7_nondense_focal_0_001_lr_300_0_0001_lr_500_0_00005_model_checkpoints_1.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_schedule')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_bands():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_bands'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 7 -l mse focal focal --lr 0.001 -m dense non non -e 1000 -s 885 0 0 -d 2022-08-05 --bands rgb lwir swir --bands rgb swir --bands rgb" 
    os.system(test_gen)

    fnames = ['3x3_dense_mse_0_001_lwir_swir_model_checkpoints_0.sh', '7x7_nondense_focal_0_001_swir_model_checkpoints_1.sh', '7x7_nondense_focal_0_001_rgb_only_model_checkpoints_2.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_bands')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_random_crop():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_random_crop'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 7 -l mse focal focal --lr 0.001 -m dense non non -e 1000 -s 885 0 0 -d 2022-08-05 --bands rgb lwir swir --bands rgb swir --bands rgb --random_crop"
    os.system(test_gen)

    fnames = ['3x3_dense_mse_0_001_lwir_swir_crop_model_checkpoints_0.sh', '7x7_nondense_focal_0_001_swir_crop_model_checkpoints_1.sh', '7x7_nondense_focal_0_001_rgb_only_crop_model_checkpoints_2.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_random_crop')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_n_cpus():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_workers'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 7 -l mse focal focal --lr 0.001 -m dense non non -e 1000 -s 885 0 0 -d 2022-08-05 --bands rgb lwir swir --bands rgb swir --bands rgb --random_crop --n_cpus 10"
    os.system(test_gen)

    fnames = ['3x3_dense_mse_0_001_lwir_swir_crop_model_checkpoints_0.sh', '7x7_nondense_focal_0_001_swir_crop_model_checkpoints_1.sh', '7x7_nondense_focal_0_001_rgb_only_crop_model_checkpoints_2.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_workers')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_width():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_width'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 7 -l mse focal focal --lr 0.001 -m dense non non -e 1000 -s 885 0 0 -d 2022-08-05 --bands rgb lwir swir --bands rgb swir --bands rgb --random_crop --n_cpus 10 --wide --width 8 2 2"
    os.system(test_gen)

    fnames = ['3x3_dense_wide_mse_0_001_lwir_swir_crop_model_checkpoints_0.sh', '7x7_nondense_wide_focal_0_001_swir_crop_model_checkpoints_1.sh', '7x7_nondense_wide_focal_0_001_rgb_only_crop_model_checkpoints_2.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_width')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_inv_type():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_inv_type'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 7 -l mse focal focal --lr 0.001 -m dense non non -e 1000 -s 885 0 0 -d 2022-08-05 --bands rgb lwir swir --bands rgb swir --bands rgb --random_crop --n_cpus 10 --wide --width 8 2 2 -i C8 NONE C8"
    os.system(test_gen)

    fnames = ['3x3_dense_wide_mse_0_001_lwir_swir_crop_C8_model_checkpoints_0.sh', '7x7_nondense_wide_focal_0_001_swir_crop_NONE_model_checkpoints_1.sh', '7x7_nondense_wide_focal_0_001_rgb_only_crop_C8_model_checkpoints_2.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_inv_type')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_comboize():
   #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_comboize'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l mse focal --lr 0.001 -m dense non -e 1000 --bands rgb lwir swir -i C8 NONE --comboize"
    os.system(test_gen)

    fnames = ['3x3_dense_mse_0_001_lwir_swir_C8_model_checkpoints_0.sh', '3x3_dense_mse_0_001_lwir_swir_NONE_model_checkpoints_1.sh', '3x3_nondense_mse_0_001_lwir_swir_C8_model_checkpoints_2.sh', '3x3_nondense_mse_0_001_lwir_swir_NONE_model_checkpoints_3.sh', '3x3_dense_focal_0_001_lwir_swir_C8_model_checkpoints_4.sh', '3x3_dense_focal_0_001_lwir_swir_NONE_model_checkpoints_5.sh', '3x3_nondense_focal_0_001_lwir_swir_C8_model_checkpoints_6.sh', '3x3_nondense_focal_0_001_lwir_swir_NONE_model_checkpoints_7.sh','7x7_dense_mse_0_001_lwir_swir_C8_model_checkpoints_8.sh', '7x7_dense_mse_0_001_lwir_swir_NONE_model_checkpoints_9.sh', '7x7_nondense_mse_0_001_lwir_swir_C8_model_checkpoints_10.sh', '7x7_nondense_mse_0_001_lwir_swir_NONE_model_checkpoints_11.sh', '7x7_dense_focal_0_001_lwir_swir_C8_model_checkpoints_12.sh', '7x7_dense_focal_0_001_lwir_swir_NONE_model_checkpoints_13.sh', '7x7_nondense_focal_0_001_lwir_swir_C8_model_checkpoints_14.sh', '7x7_nondense_focal_0_001_lwir_swir_NONE_model_checkpoints_15.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_comboize')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_gpus():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_gpus'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l mse focal --lr 0.001 -m dense non -e 1000 --bands rgb lwir swir -i C8 NONE --comboize --n_gpus 4"
    os.system(test_gen)

    fnames = ['3x3_dense_mse_0_001_lwir_swir_C8_model_checkpoints_0.sh', '3x3_dense_mse_0_001_lwir_swir_NONE_model_checkpoints_1.sh', '3x3_nondense_mse_0_001_lwir_swir_C8_model_checkpoints_2.sh', '3x3_nondense_mse_0_001_lwir_swir_NONE_model_checkpoints_3.sh', '3x3_dense_focal_0_001_lwir_swir_C8_model_checkpoints_4.sh', '3x3_dense_focal_0_001_lwir_swir_NONE_model_checkpoints_5.sh', '3x3_nondense_focal_0_001_lwir_swir_C8_model_checkpoints_6.sh', '3x3_nondense_focal_0_001_lwir_swir_NONE_model_checkpoints_7.sh','7x7_dense_mse_0_001_lwir_swir_C8_model_checkpoints_8.sh', '7x7_dense_mse_0_001_lwir_swir_NONE_model_checkpoints_9.sh', '7x7_nondense_mse_0_001_lwir_swir_C8_model_checkpoints_10.sh', '7x7_nondense_mse_0_001_lwir_swir_NONE_model_checkpoints_11.sh', '7x7_dense_focal_0_001_lwir_swir_C8_model_checkpoints_12.sh', '7x7_dense_focal_0_001_lwir_swir_NONE_model_checkpoints_13.sh', '7x7_nondense_focal_0_001_lwir_swir_C8_model_checkpoints_14.sh', '7x7_nondense_focal_0_001_lwir_swir_NONE_model_checkpoints_15.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_gpus')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_road_dataset():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_roads'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l mse focal --lr 0.001 -m dense non -e 1000 --bands rgb -i C8 NONE --comboize --n_gpus 4 --dataset road"
    os.system(test_gen)

    fnames = ['3x3_dense_mse_0_001_rgb_only_C8_road_model_checkpoints_0.sh', '3x3_dense_mse_0_001_rgb_only_NONE_road_model_checkpoints_1.sh', '3x3_nondense_mse_0_001_rgb_only_C8_road_model_checkpoints_2.sh', '3x3_nondense_mse_0_001_rgb_only_NONE_road_model_checkpoints_3.sh', '3x3_dense_focal_0_001_rgb_only_C8_road_model_checkpoints_4.sh', '3x3_dense_focal_0_001_rgb_only_NONE_road_model_checkpoints_5.sh', '3x3_nondense_focal_0_001_rgb_only_C8_road_model_checkpoints_6.sh', '3x3_nondense_focal_0_001_rgb_only_NONE_road_model_checkpoints_7.sh','7x7_dense_mse_0_001_rgb_only_C8_road_model_checkpoints_8.sh', '7x7_dense_mse_0_001_rgb_only_NONE_road_model_checkpoints_9.sh', '7x7_nondense_mse_0_001_rgb_only_C8_road_model_checkpoints_10.sh', '7x7_nondense_mse_0_001_rgb_only_NONE_road_model_checkpoints_11.sh', '7x7_dense_focal_0_001_rgb_only_C8_road_model_checkpoints_12.sh', '7x7_dense_focal_0_001_rgb_only_NONE_road_model_checkpoints_13.sh', '7x7_nondense_focal_0_001_rgb_only_C8_road_model_checkpoints_14.sh', '7x7_nondense_focal_0_001_rgb_only_NONE_road_model_checkpoints_15.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_roads')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_shadow_dataset():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_shadows'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l focal --lr 0.001 -m dense non -e 1000 --bands rgb -i C8 NONE --comboize --n_gpus 4 --dataset shadow"
    os.system(test_gen)

    fnames = ['3x3_dense_focal_0_001_rgb_only_C8_shadow_model_checkpoints_0.sh', '3x3_dense_focal_0_001_rgb_only_NONE_shadow_model_checkpoints_1.sh', '3x3_nondense_focal_0_001_rgb_only_C8_shadow_model_checkpoints_2.sh', '3x3_nondense_focal_0_001_rgb_only_NONE_shadow_model_checkpoints_3.sh', '7x7_dense_focal_0_001_rgb_only_C8_shadow_model_checkpoints_4.sh', '7x7_dense_focal_0_001_rgb_only_NONE_shadow_model_checkpoints_5.sh', '7x7_nondense_focal_0_001_rgb_only_C8_shadow_model_checkpoints_6.sh', '7x7_nondense_focal_0_001_rgb_only_NONE_shadow_model_checkpoints_7.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_shadows')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_multiclass_dataset():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_multiclass'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l focal --lr 0.001 -m dense non -e 1000 --bands rgb -i C8 NONE --comboize --n_gpus 4 --dataset clouds_and_shadow"
    os.system(test_gen)

    fnames = ['3x3_dense_focal_0_001_rgb_only_C8_clouds_and_shadow_model_checkpoints_0.sh', '3x3_dense_focal_0_001_rgb_only_NONE_clouds_and_shadow_model_checkpoints_1.sh', '3x3_nondense_focal_0_001_rgb_only_C8_clouds_and_shadow_model_checkpoints_2.sh', '3x3_nondense_focal_0_001_rgb_only_NONE_clouds_and_shadow_model_checkpoints_3.sh', '7x7_dense_focal_0_001_rgb_only_C8_clouds_and_shadow_model_checkpoints_4.sh', '7x7_dense_focal_0_001_rgb_only_NONE_clouds_and_shadow_model_checkpoints_5.sh', '7x7_nondense_focal_0_001_rgb_only_C8_clouds_and_shadow_model_checkpoints_6.sh', '7x7_nondense_focal_0_001_rgb_only_NONE_clouds_and_shadow_model_checkpoints_7.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_multiclass')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()


def test_class_weighted_loss():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_class_weighted_loss'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l cross_entropy focal --lr 0.001 -m dense non -e 1000 --bands rgb -i C8 NONE --comboize --n_gpus 4 --dataset road --class_weighted_loss"
    os.system(test_gen)

    fnames = ['3x3_dense_cross_entropy_0_001_rgb_only_C8_road_weighted_model_checkpoints_0.sh', '3x3_dense_cross_entropy_0_001_rgb_only_NONE_road_weighted_model_checkpoints_1.sh', '3x3_nondense_cross_entropy_0_001_rgb_only_C8_road_weighted_model_checkpoints_2.sh', '3x3_nondense_cross_entropy_0_001_rgb_only_NONE_road_weighted_model_checkpoints_3.sh', '3x3_dense_focal_0_001_rgb_only_C8_road_weighted_model_checkpoints_4.sh', '3x3_dense_focal_0_001_rgb_only_NONE_road_weighted_model_checkpoints_5.sh', '3x3_nondense_focal_0_001_rgb_only_C8_road_weighted_model_checkpoints_6.sh', '3x3_nondense_focal_0_001_rgb_only_NONE_road_weighted_model_checkpoints_7.sh','7x7_dense_cross_entropy_0_001_rgb_only_C8_road_weighted_model_checkpoints_8.sh', '7x7_dense_cross_entropy_0_001_rgb_only_NONE_road_weighted_model_checkpoints_9.sh', '7x7_nondense_cross_entropy_0_001_rgb_only_C8_road_weighted_model_checkpoints_10.sh', '7x7_nondense_cross_entropy_0_001_rgb_only_NONE_road_weighted_model_checkpoints_11.sh', '7x7_dense_focal_0_001_rgb_only_C8_road_weighted_model_checkpoints_12.sh', '7x7_dense_focal_0_001_rgb_only_NONE_road_weighted_model_checkpoints_13.sh', '7x7_nondense_focal_0_001_rgb_only_C8_road_weighted_model_checkpoints_14.sh', '7x7_nondense_focal_0_001_rgb_only_NONE_road_weighted_model_checkpoints_15.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_class_weighted_loss')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_class_weighted_loss_explicit():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_class_weighted_loss_explicit'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l cross_entropy focal --lr 0.001 -m dense non -e 1000 --bands rgb -i C8 NONE --comboize --n_gpus 4 --dataset road --class_weighted_loss --weights 0.2 0.8"
    os.system(test_gen)

    fnames = ['3x3_dense_cross_entropy_0_001_rgb_only_C8_road_weighted_0_2_0_8_model_checkpoints_0.sh', '3x3_dense_cross_entropy_0_001_rgb_only_NONE_road_weighted_0_2_0_8_model_checkpoints_1.sh', '3x3_nondense_cross_entropy_0_001_rgb_only_C8_road_weighted_0_2_0_8_model_checkpoints_2.sh', '3x3_nondense_cross_entropy_0_001_rgb_only_NONE_road_weighted_0_2_0_8_model_checkpoints_3.sh', '3x3_dense_focal_0_001_rgb_only_C8_road_weighted_0_2_0_8_model_checkpoints_4.sh', '3x3_dense_focal_0_001_rgb_only_NONE_road_weighted_0_2_0_8_model_checkpoints_5.sh', '3x3_nondense_focal_0_001_rgb_only_C8_road_weighted_0_2_0_8_model_checkpoints_6.sh', '3x3_nondense_focal_0_001_rgb_only_NONE_road_weighted_0_2_0_8_model_checkpoints_7.sh','7x7_dense_cross_entropy_0_001_rgb_only_C8_road_weighted_0_2_0_8_model_checkpoints_8.sh', '7x7_dense_cross_entropy_0_001_rgb_only_NONE_road_weighted_0_2_0_8_model_checkpoints_9.sh', '7x7_nondense_cross_entropy_0_001_rgb_only_C8_road_weighted_0_2_0_8_model_checkpoints_10.sh', '7x7_nondense_cross_entropy_0_001_rgb_only_NONE_road_weighted_0_2_0_8_model_checkpoints_11.sh', '7x7_dense_focal_0_001_rgb_only_C8_road_weighted_0_2_0_8_model_checkpoints_12.sh', '7x7_dense_focal_0_001_rgb_only_NONE_road_weighted_0_2_0_8_model_checkpoints_13.sh', '7x7_nondense_focal_0_001_rgb_only_C8_road_weighted_0_2_0_8_model_checkpoints_14.sh', '7x7_nondense_focal_0_001_rgb_only_NONE_road_weighted_0_2_0_8_model_checkpoints_15.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_class_weighted_loss_explicit')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_iou_batch_size_gradient_accumulation():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_iou'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l iou soft_iou --lr 1e-3 -m dense non -e 1000 --bands rgb -i C8 NONE --comboize --n_gpus 4 --dataset road --class_weighted_loss --weights 0.2 0.8 --batch_size 8 --num_accumulation_steps 4"
    os.system(test_gen)

    fnames = ['3x3_dense_iou_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_0.sh', '3x3_dense_iou_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_1.sh', '3x3_nondense_iou_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_2.sh', '3x3_nondense_iou_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_3.sh', '3x3_dense_soft_iou_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_4.sh', '3x3_dense_soft_iou_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_5.sh', '3x3_nondense_soft_iou_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_6.sh', '3x3_nondense_soft_iou_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_7.sh','7x7_dense_iou_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_8.sh', '7x7_dense_iou_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_9.sh', '7x7_nondense_iou_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_10.sh', '7x7_nondense_iou_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_11.sh', '7x7_dense_soft_iou_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_12.sh', '7x7_dense_soft_iou_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_13.sh', '7x7_nondense_soft_iou_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_14.sh', '7x7_nondense_soft_iou_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_15.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_iou')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_jaccard():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_jaccard'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l jaccard log_jaccard --lr 1e-3 -m dense non -e 1000 --bands rgb -i C8 NONE --comboize --n_gpus 4 --dataset road --class_weighted_loss --weights 0.2 0.8 --batch_size 8 --num_accumulation_steps 4"
    os.system(test_gen)

    fnames = ['3x3_dense_jaccard_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_0.sh', '3x3_dense_jaccard_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_1.sh', '3x3_nondense_jaccard_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_2.sh', '3x3_nondense_jaccard_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_3.sh', '3x3_dense_log_jaccard_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_4.sh', '3x3_dense_log_jaccard_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_5.sh', '3x3_nondense_log_jaccard_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_6.sh', '3x3_nondense_log_jaccard_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_7.sh','7x7_dense_jaccard_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_8.sh', '7x7_dense_jaccard_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_9.sh', '7x7_nondense_jaccard_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_10.sh', '7x7_nondense_jaccard_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_11.sh', '7x7_dense_log_jaccard_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_12.sh', '7x7_dense_log_jaccard_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_13.sh', '7x7_nondense_log_jaccard_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_14.sh', '7x7_nondense_log_jaccard_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_15.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_jaccard')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_dice():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_dice'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l dice log_dice --lr 1e-3 -m dense non -e 1000 --bands rgb -i C8 NONE --comboize --n_gpus 4 --dataset road --class_weighted_loss --weights 0.2 0.8 --batch_size 8 --num_accumulation_steps 4"
    os.system(test_gen)

    fnames = ['3x3_dense_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_0.sh', '3x3_dense_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_1.sh', '3x3_nondense_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_2.sh', '3x3_nondense_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_3.sh', '3x3_dense_log_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_4.sh', '3x3_dense_log_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_5.sh', '3x3_nondense_log_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_6.sh', '3x3_nondense_log_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_7.sh','7x7_dense_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_8.sh', '7x7_dense_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_9.sh', '7x7_nondense_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_10.sh', '7x7_nondense_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_11.sh', '7x7_dense_log_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_12.sh', '7x7_dense_log_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_13.sh', '7x7_nondense_log_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_14.sh', '7x7_nondense_log_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_15.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_dice')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

def test_max_epoch():
    #Get sbatch dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_dir = f'{current_dir}/test_artifacts/test_sbatch_files_max_epoch'
    os.makedirs(sbatch_dir, exist_ok=True)

    #Generate sbatch files
    test_gen = f"python3 ../scripts/generate_sbatch.py -o {sbatch_dir} -k 3 7 -l dice log_dice --lr 1e-3 -m dense non -e 1000 --bands rgb -i C8 NONE --comboize --n_gpus 4 --dataset road --class_weighted_loss --weights 0.2 0.8 --batch_size 8 --num_accumulation_steps 4 --max_epoch 1500"
    os.system(test_gen)

    fnames = ['3x3_dense_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_0.sh', '3x3_dense_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_1.sh', '3x3_nondense_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_2.sh', '3x3_nondense_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_3.sh', '3x3_dense_log_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_4.sh', '3x3_dense_log_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_5.sh', '3x3_nondense_log_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_6.sh', '3x3_nondense_log_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_7.sh','7x7_dense_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_8.sh', '7x7_dense_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_9.sh', '7x7_nondense_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_10.sh', '7x7_nondense_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_11.sh', '7x7_dense_log_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_12.sh', '7x7_dense_log_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_13.sh', '7x7_nondense_log_dice_0_001_rgb_only_C8_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_14.sh', '7x7_nondense_log_dice_0_001_rgb_only_NONE_road_weighted_0_2_0_8_batch_size_8_nsteps_4_model_checkpoints_15.sh']
    ref_dir = os.path.join(current_dir, 'test_permanent/test_sbatch_files_max_epoch')
    for fname in fnames:
        assert os.path.exists(os.path.join(sbatch_dir, fname))
        with open(os.path.join(sbatch_dir, fname), 'r') as f:
            ref_file = open(os.path.join(ref_dir, fname), 'r')
            #Last 10 chars are current date and dont have to match
            assert f.read()[:-10] == ref_file.read()[:-10]
            ref_file.close()

