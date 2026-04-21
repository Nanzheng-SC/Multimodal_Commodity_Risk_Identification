import os
import torch
from custom.models import TimeMixer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimeMixer': TimeMixer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            import platform
            if platform.system() == 'Darwin':
                device = torch.device('mps')
                print('Use MPS')
                return device
            # 检查CUDA是否可用
            if torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                if self.args.use_multi_gpu:
                    print('Use GPU: cuda{}'.format(self.args.device_ids))
                else:
                    print('Use GPU: cuda:{}'.format(self.args.gpu))
                print(f'GPU memory: {torch.cuda.get_device_properties(self.args.gpu).total_memory / 1e9:.2f} GB')
            else:
                # 保持设备选择与运行配置一致，便于在统一环境中复现实验。
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('CUDA not available, but forcing GPU usage')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
