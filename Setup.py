import os
import urllib.request
import zipfile

# Klasörleri oluştur
os.makedirs('nnUNet_results', exist_ok=True)
os.makedirs('nnUNet_raw', exist_ok=True)
os.makedirs('nnUNet_preprocessed', exist_ok=True)
os.makedirs('ct_input', exist_ok=True)
os.makedirs('ct_output', exist_ok=True)
os.makedirs('damar_output', exist_ok=True)

os.environ['RESULTS_FOLDER'] = os.path.abspath('nnUNet_results')
os.environ['nnUNet_raw_data_base'] = os.path.abspath('nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.abspath('nnUNet_preprocessed')

# 6 CT task
tasks = {
    "003": "Task003_Liver",
    "006": "Task006_Lung",
    "007": "Task007_Pancreas",
    "008": "Task008_HepaticVessel",
    "009": "Task009_Spleen",
    "010": "Task010_Colon"
}

for task_id, task_name in tasks.items():
    zip_path = f"{task_name}.zip"
    url = f"https://zenodo.org/records/4485926/files/{task_name}.zip"

    print(f"\n{task_name} indiriliyor...")
    urllib.request.urlretrieve(url, zip_path,
                               reporthook=lambda b, bs, total: print(
                                   f"\r{b * bs / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB", end=''))

    print(f"\nAçılıyor...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall('nnUNet_results')

    os.remove(zip_path)
    print(f"✅ {task_name} tamamlandı")

print("\n🎉 Tüm modeller indirildi!")