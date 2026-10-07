import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from src.dataset import HAM10000Dataset
from src.metrics import binary_metrics
from src.cross_attention import CrossAttentionModel


def load_backbone(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    prefix = 'image_encoder.backbone.'
    state = {k[len(prefix):]: v for k, v in ckpt['model_state'].items() if k.startswith(prefix)}
    if not state:
        raise RuntimeError('No image_encoder.backbone.* weights found.')
    missing, unexpected = model.image_encoder.backbone.load_state_dict(state, strict=False)
    print(f'Initialized image backbone from {checkpoint_path}')
    if missing: print('Missing:', missing)
    if unexpected: print('Unexpected:', unexpected)


def forward(model, batch, device):
    return model(batch['image'].to(device), batch['age_z'].to(device),
                 batch['sex_idx'].to(device), batch['localization_idx'].to(device))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    for batch in loader:
        logits = forward(model, batch, device)
        ys.extend(batch['target'].numpy().tolist())
        ps.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return binary_metrics(ys, ps, threshold=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/manifest.csv')
    ap.add_argument('--schema', default='data/tabular_schema.json')
    ap.add_argument('--init-image-checkpoint', default='outputs/image_best.pt')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output-dir', default='outputs')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    schema = json.load(open(args.schema))
    model = CrossAttentionModel(len(schema['sex_categories']), len(schema['localization_categories']), pretrained=False).to(device)
    load_backbone(model, args.init_image_checkpoint)

    train_ds = HAM10000Dataset(args.manifest, 'train', args.image_size, True)
    val_ds = HAM10000Dataset(args.manifest, 'val', args.image_size, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type=='cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type=='cuda')

    df = pd.read_csv(args.manifest); tr = df[df.split=='train']
    n_pos = int(tr.target.sum()); n_neg = len(tr)-n_pos; pos_weight=n_neg/max(n_pos,1)
    print(f'Train positives={n_pos}, negatives={n_neg}, pos_weight={pos_weight:.3f}')
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out = Path(args.output_dir); out.mkdir(exist_ok=True)
    best=-1; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in train_loader:
            y=batch['target'].to(device); opt.zero_grad(); logits=forward(model,batch,device)
            loss=loss_fn(logits,y); loss.backward(); opt.step(); losses.append(loss.item())
        m=evaluate(model,val_loader,device)
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),**{f'val_{k}':v for k,v in m.items() if isinstance(v,(int,float))}})
        print(f"Epoch {epoch:02d} | train_loss={np.mean(losses):.4f} | val_AUROC={m['auroc']:.4f} | val_AUPRC={m['auprc']:.4f} | val_F1={m['f1']:.4f}")
        if m['auroc']>best:
            best=m['auroc']
            torch.save({'model_name':'cross','model_state':model.state_dict(),'schema':schema,'image_size':args.image_size,'best_val_auroc':best}, out/'cross_best.pt')
            print('  -> saved new best checkpoint: outputs/cross_best.pt')
    json.dump(history, open(out/'cross_history.json','w'), indent=2)
    print(f'Best validation AUROC: {best:.4f}')

if __name__=='__main__': main()
