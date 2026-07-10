import subprocess, os

env = os.environ.copy()
env['RESULTS_FOLDER'] = r'C:\Users\ASUS\PycharmProjects\TemelProje\nnUNet_results'
env['nnUNet_raw_data_base'] = r'C:\Users\ASUS\PycharmProjects\TemelProje\nnUNet_raw'
env['nnUNet_preprocessed'] = r'C:\Users\ASUS\PycharmProjects\TemelProje\nnUNet_preprocessed'

os.makedirs(r'C:\Users\ASUS\PycharmProjects\TemelProje\ct_input', exist_ok=True)
os.makedirs(r'C:\Users\ASUS\PycharmProjects\TemelProje\ct_output', exist_ok=True)

result = subprocess.run([
    r'C:\Users\ASUS\PycharmProjects\TemelProje\radyoloji_env_311\Scripts\nnUNet_predict.exe',
    '-i', r'C:\Users\ASUS\PycharmProjects\TemelProje\ct_input',
    '-o', r'C:\Users\ASUS\PycharmProjects\TemelProje\ct_output',
    '-t', '003',
    '-m', '3d_fullres',
    '-f', '0', '1', '2', '3', '4',
    '--overwrite_existing'
], env=env, capture_output=True, text=True)

print('STDOUT:', result.stdout[-2000:])
print('STDERR:', result.stderr[-2000:])
print('Return code:', result.returncode)
