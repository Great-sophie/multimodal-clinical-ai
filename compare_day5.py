import json
from pathlib import Path
import pandas as pd
rows=[]
for name in ['image','tabular','late','cross']:
    p=Path('outputs')/f'{name}_test_metrics.json'
    if not p.exists():
        print('Skipping missing:',p); continue
    m=json.load(open(p))
    rows.append({'model':name,'AUROC':m['auroc'],'AUPRC':m['auprc'],'F1':m['f1'],'Balanced accuracy':m['balanced_accuracy'],'Sensitivity':m['sensitivity'],'Specificity':m['specificity']})
df=pd.DataFrame(rows); print(df.to_string(index=False,float_format=lambda x:f'{x:.4f}')); df.to_csv('outputs/model_comparison_day5.csv',index=False)
print('\nSaved: outputs/model_comparison_day5.csv')
