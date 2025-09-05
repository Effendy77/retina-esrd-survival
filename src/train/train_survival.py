import argparse, os, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.deepsurv_retina import DeepSurvRetina
from src.train.losses import neg_partial_log_likelihood
from src.utils.seed import set_seed
from src.data.ukb_survival_dataset import UKBSurvivalDataset

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--weights_path', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', default='checkpoints/fold0.pth')
    return ap.parse_args()

def main():
    args = parse()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    set_seed(42)
    tform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train_ds = UKBSurvivalDataset(args.train_csv, args.img_dir, tform)
    val_ds   = UKBSurvivalDataset(args.val_csv, args.img_dir, tform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSurvRetina(weights_path=args.weights_path).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    for epoch in range(args.epochs):
        model.train(); total=0.0
        for img, time, event in train_loader:
            img, time, event = img.to(device), time.to(device), event.to(device)
            lrisk = model(img)
            loss = neg_partial_log_likelihood(lrisk, time, event)
            optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item()
        print(f'Epoch {epoch+1}: loss={total/len(train_loader):.4f}')
    torch.save(model.state_dict(), args.out)

if __name__=='__main__':
    main()
