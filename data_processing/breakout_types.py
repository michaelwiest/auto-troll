import pandas as pd
import os
import sys

in_dir = sys.argv[1]
types = ['Right', 'Left']
out_df_base = 'russian_combined_{}'

files = [os.path.join(in_dir, f) for f in os.listdir(in_dir)
         if f.lower().endswith('.csv')]
# dfs = [pd.read_csv(f) for f in files]

for type in types:
    outdir = type.lower()
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        sub = df.loc[df.account_type == type]
        sub.to_csv(os.path.join(outdir, type + '_' + os.path.basename(f)))
