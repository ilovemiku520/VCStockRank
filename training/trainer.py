# training/trainer.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

from .loss import MultiTaskLoss


class Trainer:
    def __init__(self, model, config, train_loader, val_loader=None,
                 test_loader=None, log_dir='logs'):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        self.loss_fn = MultiTaskLoss(
            ranking_loss_weight=1.0,
            vol_loss_weight=config.LAMBDA_VOL,
            regime_loss_weight=0.1,
            decomp_loss_weight=config.LAMBDA_DECOMP
        )

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0

        # 历史记录：始终包含验证指标，无验证时 append None
        self.history = {
            'train_loss': [],
            'train_rank_loss': [],
            'train_vol_loss': [],
            'val_loss': [],
            'val_rank_loss': [],
            'val_vol_loss': [],
            'learning_rate': []
        }

        self.device = config.DEVICE

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        except:
            self.writer = None

        print(f"Trainer initialized on {self.device}")
        print(f"Train samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Val samples: {len(val_loader.dataset)}")
        if test_loader:
            print(f"Test samples: {len(test_loader.dataset)}")

    def train(self, epochs=None):
        if epochs is None:
            epochs = self.config.EPOCHS
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            self.epoch = epoch
            train_metrics = self._train_epoch()
            val_metrics = {}
            if self.val_loader:
                val_metrics = self._validate()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))

            self._update_history(train_metrics, val_metrics, current_lr)
            self._print_progress(epoch, epochs, train_metrics, val_metrics, current_lr)
            self._log_to_tensorboard(epoch, train_metrics, val_metrics)

            if self._check_early_stopping(val_metrics.get('loss', train_metrics['loss'])):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

        self._load_best_model()

        if self.test_loader:
            test_metrics = self.evaluate(self.test_loader)
            print(f"\nTest Results:")
            self._print_metrics(test_metrics)

        self._save_history()
        return self.model

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        total_rank_loss = 0
        total_vol_loss = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            x = batch['x'].to(self.device)
            rank_target = batch['rank_target'].squeeze(-1).to(self.device)
            vol_target = batch['vol_target'].squeeze(-1).to(self.device)

            outputs = self.model(x)
            targets = {'rank_target': rank_target, 'vol_target': vol_target}
            loss, loss_dict = self.loss_fn(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss_dict['total_loss']
            total_rank_loss += loss_dict.get('rank_loss', 0)
            total_vol_loss += loss_dict.get('vol_loss', 0)

        n = len(self.train_loader)
        return {
            'loss': total_loss / n,
            'rank_loss': total_rank_loss / n,
            'vol_loss': total_vol_loss / n
        }

    def _validate(self):
        self.model.eval()
        total_loss = 0
        total_rank_loss = 0
        total_vol_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['x'].to(self.device)
                rank_target = batch['rank_target'].squeeze(-1).to(self.device)
                vol_target = batch['vol_target'].squeeze(-1).to(self.device)

                outputs = self.model(x)
                targets = {'rank_target': rank_target, 'vol_target': vol_target}
                loss, loss_dict = self.loss_fn(outputs, targets)

                total_loss += loss_dict['total_loss']
                total_rank_loss += loss_dict.get('rank_loss', 0)
                total_vol_loss += loss_dict.get('vol_loss', 0)

        n = len(self.val_loader)
        return {
            'loss': total_loss / n,
            'rank_loss': total_rank_loss / n,
            'vol_loss': total_vol_loss / n
        }

    def evaluate(self, loader):
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(self.device)
                t = batch['rank_target'].squeeze(-1).cpu().numpy()
                out = self.model(x)
                p = out['rank_score'].squeeze(-1).cpu().numpy()
                preds.extend(p)
                targets.extend(t)
        preds = np.array(preds)
        targets = np.array(targets)
        ic = np.corrcoef(preds, targets)[0, 1] if len(preds) > 0 else 0
        return {
            'ic': ic,
            'mean_pred': np.mean(preds) if len(preds) else 0,
            'std_pred': np.std(preds) if len(preds) else 0,
            'mean_target': np.mean(targets) if len(targets) else 0
        }

    def _update_history(self, train_metrics, val_metrics, lr):
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_rank_loss'].append(train_metrics.get('rank_loss', 0))
        self.history['train_vol_loss'].append(train_metrics.get('vol_loss', 0))
        self.history['learning_rate'].append(lr)

        if val_metrics:
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_rank_loss'].append(val_metrics.get('rank_loss', 0))
            self.history['val_vol_loss'].append(val_metrics.get('vol_loss', 0))
        else:
            self.history['val_loss'].append(None)
            self.history['val_rank_loss'].append(None)
            self.history['val_vol_loss'].append(None)

    def _print_progress(self, epoch, epochs, train_metrics, val_metrics, lr):
        msg = f"Epoch {epoch+1}/{epochs} | Loss: {train_metrics['loss']:.4f} | Rank Loss: {train_metrics.get('rank_loss', 0):.4f} | Vol Loss: {train_metrics.get('vol_loss', 0):.4f}"
        if val_metrics:
            msg += f" | Val Loss: {val_metrics['loss']:.4f}"
        msg += f" | LR: {lr:.2e}"
        print(msg)

    def _print_metrics(self, metrics):
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    def _check_early_stopping(self, current_loss):
        if current_loss < self.best_val_loss:
            self.best_val_loss = current_loss
            self.patience_counter = 0
            self._save_checkpoint('best')
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.PATIENCE:
                return True
        return False

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config,
            'input_dim': self.config.INPUT_DIM,
        }
        fname = f'checkpoint_epoch_{epoch+1}.pt' if isinstance(epoch, int) else 'best_model.pt'
        torch.save(checkpoint, os.path.join(self.log_dir, fname))

    def _load_best_model(self):
        path = os.path.join(self.log_dir, 'best_model.pt')
        if os.path.exists(path):
            checkpoint = torch.load(path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            ep = checkpoint.get('epoch')
            if isinstance(ep, int):
                print(f"Loaded best model from epoch {ep+1}")
            else:
                print("Loaded best model (best checkpoint)")

    def _save_history(self):
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.log_dir, 'training_history.csv'), index=False)
        with open(os.path.join(self.log_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

    def _log_to_tensorboard(self, epoch, train_metrics, val_metrics):
        if self.writer is None:
            return
        self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        self.writer.add_scalar('Loss/train_rank', train_metrics.get('rank_loss', 0), epoch)
        self.writer.add_scalar('Loss/train_vol', train_metrics.get('vol_loss', 0), epoch)
        if val_metrics:
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val_rank', val_metrics.get('rank_loss', 0), epoch)
            self.writer.add_scalar('Loss/val_vol', val_metrics.get('vol_loss', 0), epoch)
        self.writer.add_scalar('Learning_rate', self.history['learning_rate'][-1], epoch)


class RollingWindowTrainer(Trainer):
    def __init__(self, model, config, data_generator, window_size=720,
                 step_size=30, **kwargs):
        super().__init__(model, config, **kwargs)
        self.data_generator = data_generator
        self.window_size = window_size
        self.step_size = step_size
        self.window_results = []

    def train_rolling(self, total_days=None):
        if total_days is None:
            total_days = self.window_size + 10 * self.step_size
        print(f"\n{'='*60}\nRolling Window Training\nWindow: {self.window_size} days, Step: {self.step_size} days\n{'='*60}\n")

        for start_day in range(0, total_days - self.window_size, self.step_size):
            print(f"\nWindow {start_day//self.step_size + 1}")
            train_end = start_day + self.window_size
            val_start, val_end = train_end, train_end + 30
            test_start, test_end = val_end, val_end + 30

            train_data, val_data, test_data = self.data_generator(
                start_day, train_end, val_start, val_end, test_start, test_end
            )

            self.train_loader = DataLoader(train_data, batch_size=self.config.BATCH_SIZE, shuffle=True)
            self.val_loader = DataLoader(val_data, batch_size=self.config.BATCH_SIZE, shuffle=False)

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

            self.model = self.train()

            if test_data:
                test_loader = DataLoader(test_data, batch_size=self.config.BATCH_SIZE, shuffle=False)
                test_metrics = self.evaluate(test_loader)
                self.window_results.append({'window': start_day//self.step_size + 1, 'test_metrics': test_metrics})
                print(f"Test IC: {test_metrics['ic']:.4f}")

        self._summarize_windows()
        return self.model

    def _summarize_windows(self):
        print(f"\n{'='*60}\nRolling Window Results Summary\n{'='*60}")
        ics = [r['test_metrics']['ic'] for r in self.window_results]
        if ics:
            print(f"Average IC: {np.mean(ics):.4f}")
            print(f"IC Std: {np.std(ics):.4f}")
            print(f"ICIR: {np.mean(ics)/(np.std(ics)+1e-8):.4f}")
            print(f"Win Rate: {np.mean(np.array(ics)>0):.2%}")
        df = pd.DataFrame(self.window_results)
        df.to_csv(os.path.join(self.log_dir, 'rolling_window_results.csv'), index=False)
        return df