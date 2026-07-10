import nnunet

preproc_file = nnunet.__file__.replace('__init__.py', '') + 'preprocessing/preprocessing.py'

with open(preproc_file, 'r') as f:
    content = f.read()

# mgrid'i int32'ye çevir
content = content.replace(
    'map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]',
    'map_rows, map_cols, map_dims = [x.astype(np.int32) for x in np.mgrid[:rows, :cols, :dim]]'
)

with open(preproc_file, 'w') as f:
    f.write(content)

print("✅ Düzeltildi!")