import os
import nnunet
from nnunet.inference.predict import predict_from_folder

os.environ['RESULTS_FOLDER'] = os.path.abspath('nnUNet_results')
os.environ['nnUNet_raw_data_base'] = os.path.abspath('nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.abspath('nnUNet_preprocessed')

restore_file = nnunet.__file__.replace('__init__.py', '') + 'training/model_restore.py'
with open(restore_file, 'r') as f:
    content = f.read()
content = content.replace(
    "torch.load(i, map_location=torch.device('cpu'))",
    "torch.load(i, map_location=torch.device('cpu'), weights_only=False)"
)
with open(restore_file, 'w') as f:
    f.write(content)

if __name__ == '__main__':
    os.makedirs('ct_output', exist_ok=True)

    model_folder = os.path.abspath('nnUNet_results/nnUNet/3d_fullres/Task007_Pancreas/nnUNetTrainerV2__nnUNetPlansv2.1')

    predict_from_folder(
        model_folder,
        os.path.abspath('ct_input'),
        os.path.abspath('ct_output'),
        folds=[0, 1, 2, 3, 4],
        save_npz=False,
        num_threads_preprocessing=1,
        num_threads_nifti_save=1,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=True,
        mixed_precision=True,
        overwrite_existing=True
    )

    print("✅ Inference tamamlandı!")