import torch
from torch.utils.data import DataLoader

class pretrain_evaluator:
    """
    The evaluation functions for the pretraining models.
    """
    
    def __init__(self, device='cuda'):
        self.device = device

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader):
        total_loss = 0.
        model.eval()  # turn on evaluation mode

        with torch.no_grad():
            for samples in dataloader:
                samples = samples.to(self.device, non_blocking=True, dtype=torch.float)
                loss, _, _ = model(samples)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate_spe(self, model, spe_arr):
        """
        Use the mae model to evaluate the spectra directly.
        Input: mae-like model, spetra in numpy array
        Output: loss, pred, mask
        """
        model.eval()
        with torch.no_grad():
            torch.manual_seed(24)
            spe = torch.from_numpy(spe_arr).to(self.device, non_blocking=True, dtype=torch.float)

            return model(spe.unsqueeze(0))

    def get_mse(self, true_values, pred_values):
        # all in tensor
        loss = (true_values - pred_values) ** 2
        loss = loss.mean()

        return loss.item()
    
    def evaluate_inraw(self, model, dataloader_raw, dataloader_norm, inverse=None, device='cuda'):
        """
        dataloaders: the shuffling should be turned off for both dataloaders, otherwise the indices won't be matched.
                    the validation dataloader from get_dataloader() has shuffling turned off by default.
        inverse: function to inverse the normalized image to raw image
                inverse_standardize or inverse_log
        """
        total_loss = 0.
        model.eval()  # turn on evaluation mode

        with torch.no_grad():
            for (raws, norms) in zip(dataloader_raw, dataloader_norm):
                raws = raws.to(device, non_blocking=True, dtype=torch.float)
                norms = norms.to(device, non_blocking=True, dtype=torch.float)

                _, pred, _ = model(norms)    # in normalized space
                pred_un = model.unpatchify(pred) # the mae model has unpatchify function 
                if inverse:
                    pred_un = inverse(raws, pred_un)    # in raw space
                loss = self.get_mse(raws, pred_un)
                total_loss += loss

        return total_loss / len(dataloader_raw)

    def evaluate_base(self, dataloader):
        data = torch.empty((0, 2048))

        with torch.no_grad():
            for samples in dataloader:
                data = torch.cat((data, samples), 0)
            mean = data.mean()
            mse = self.get_mse(data, mean)
            
        return mse

    def inverse_standardize(self, raws, pred_un):
        mean = raws.mean(dim=1, keepdim=True)
        std = raws.std(dim=1, keepdim=True)

        return pred_un * std + mean

    def inverse_log(self, raws, pred_un):
        # raws is not used, just to be consistent with other inverse functions
        return torch.exp(pred_un) - 1


class finetune_evaluator:
    """
    The evaluation functions for the finetuning models.
    """
    
    def __init__(self, device='cuda'):
        self.device = device

    def get_mse(self, true_values, pred_values):
        # all in tensor
        loss = (true_values - pred_values) ** 2
        loss = loss.mean(dim=0)
        return loss.cpu().numpy()

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader):
        total_loss = 0.
        model.eval()  # turn on evaluation mode

        with torch.no_grad():
            for batch in dataloader:
                samples = batch['spe'].to(self.device, non_blocking=True, dtype=torch.float)
                targets = batch['target'].to(self.device, non_blocking=True, dtype=torch.float)

                preds, _ = model(samples)
                loss = self.get_mse(targets, preds)
                total_loss += loss

        return total_loss / len(dataloader)
    
    def evaluate_base(self, dataloader):
        total_loss = 0.

        with torch.no_grad():
            for batch in dataloader:
                mean = batch['target'].mean(dim=0)
                loss = self.get_mse(batch['target'], mean)
                total_loss += loss

        return total_loss / len(dataloader)
    
def main():
    import sys
    sys.path.append('/workspaces/xpt')
    from models_regressor import mae_vit_base_patch16
    from util.datasets import get_dataloader, standardize

    model = mae_vit_base_patch16(pretrained=True).to('cuda')
    checkpoint = torch.load('models/mae_base_patch16_lr_1e-06_20240115.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    dataloader = get_dataloader(ispretrain=False, annotations_file='info_20240112.csv', input_dir='data/finetune/train', 
                                batch_size=256, transform=standardize)

    eva = finetune_evaluator()
    loss = eva.evaluate(model=model, dataloader=dataloader['val'])
    print(loss)

    print(eva.evaluate_base(dataloader['val']))

if __name__ == "__main__":
    main()