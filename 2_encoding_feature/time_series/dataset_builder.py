import numpy as np
from .config import TIME_SERIES_CONFIG

class DatasetBuilder:
    def __init__(self):
        self.split_strategy = TIME_SERIES_CONFIG.get('split_strategy', 'ratio')
        self.fixed_valid_size = TIME_SERIES_CONFIG.get('fixed_valid_size', 12)
        self.fixed_test_size = TIME_SERIES_CONFIG.get('fixed_test_size', 12)
        self.train_ratio = TIME_SERIES_CONFIG['train_ratio']
        self.valid_ratio = TIME_SERIES_CONFIG['valid_ratio']
        self.test_ratio = TIME_SERIES_CONFIG['test_ratio']
    
    def split_dataset(self, X, y, months):
        total_samples = len(X)

        if self.split_strategy == 'fixed_horizon':
            valid_size = min(self.fixed_valid_size, max(total_samples // 4, 1))
            test_size = min(self.fixed_test_size, max(total_samples // 4, 1))
            if total_samples > valid_size + test_size:
                train_end = total_samples - valid_size - test_size
                valid_end = total_samples - test_size
            else:
                train_end = int(total_samples * self.train_ratio)
                valid_end = int(total_samples * (self.train_ratio + self.valid_ratio))
        else:
            train_end = int(total_samples * self.train_ratio)
            valid_end = int(total_samples * (self.train_ratio + self.valid_ratio))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        months_train = months[:train_end]
        
        X_valid = X[train_end:valid_end]
        y_valid = y[train_end:valid_end]
        months_valid = months[train_end:valid_end]
        
        X_test = X[valid_end:]
        y_test = y[valid_end:]
        months_test = months[valid_end:]
        
        return {
            'train': {
                'X': X_train,
                'y': y_train,
                'months': months_train
            },
            'valid': {
                'X': X_valid,
                'y': y_valid,
                'months': months_valid
            },
            'test': {
                'X': X_test,
                'y': y_test,
                'months': months_test
            }
        }
    
    def save_dataset(self, dataset, output_dir, window_length):
        import os
        
        window_dir = os.path.join(output_dir, f'window_{window_length}')
        os.makedirs(window_dir, exist_ok=True)
        
        np.save(os.path.join(window_dir, 'train.npy'), dataset['train']['X'])
        np.save(os.path.join(window_dir, 'train_labels.npy'), dataset['train']['y'])
        np.save(os.path.join(window_dir, 'train_months.npy'), dataset['train']['months'])
        np.save(os.path.join(window_dir, 'train_index.npy'), dataset['train']['months'])
        
        np.save(os.path.join(window_dir, 'valid.npy'), dataset['valid']['X'])
        np.save(os.path.join(window_dir, 'valid_labels.npy'), dataset['valid']['y'])
        np.save(os.path.join(window_dir, 'valid_months.npy'), dataset['valid']['months'])
        np.save(os.path.join(window_dir, 'valid_index.npy'), dataset['valid']['months'])
        
        np.save(os.path.join(window_dir, 'test.npy'), dataset['test']['X'])
        np.save(os.path.join(window_dir, 'test_labels.npy'), dataset['test']['y'])
        np.save(os.path.join(window_dir, 'test_months.npy'), dataset['test']['months'])
        np.save(os.path.join(window_dir, 'test_index.npy'), dataset['test']['months'])
        
        return window_dir
