import argparse, json
from pathlib import Path
import pandas as pd, torch
from torch.utils.data import DataLoader
from src.dataset import HAM10000Dataset
from src.metrics import binary_metrics
from src.cross_attention import CrossAttentionModel

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default='outputs/cross_best.pt'); ap.add_argument('--manifest',default='data/manifest.csv'); ap.add_argument('--split',default='test'); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--num-workers',type=int,default=4); ap.add_argument('--output-dir',default='outputs'); args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(args.checkpoint,map_location='cpu'); s=ckpt['schema']
    model=CrossAttentionModel(len(s['sex_categories']),len(s['localization_categories']),pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state']); model.eval()
    ds=HAM10000Dataset(args.manifest,args.split,ckpt.get('image_size',224),False)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=device.type=='cuda')
    ys=[]; ps=[]; ids=[]; lesions=[]
    for b in dl:
        logits=model(b['image'].to(device),b['age_z'].to(device),b['sex_idx'].to(device),b['localization_idx'].to(device))
        ys.extend(b['target'].numpy().tolist()); ps.extend(torch.sigmoid(logits).cpu().numpy().tolist()); ids.extend(b['image_id']); lesions.extend(b['lesion_id'])
    m=binary_metrics(ys,ps,0.5); print(json.dumps(m,indent=2))
    out=Path(args.output_dir); out.mkdir(exist_ok=True)
    json.dump(m,open(out/f'cross_{args.split}_metrics.json','w'),indent=2)
    pd.DataFrame({'image_id':ids,'lesion_id':lesions,'y_true':ys,'y_prob':ps}).to_csv(out/f'cross_{args.split}_predictions.csv',index=False)
    print(f'Saved: {out/f"cross_{args.split}_metrics.json"}')
    print(f'Saved: {out/f"cross_{args.split}_predictions.csv"}')

if __name__=='__main__': main()
