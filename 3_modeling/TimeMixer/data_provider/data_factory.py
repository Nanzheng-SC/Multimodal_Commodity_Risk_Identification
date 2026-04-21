from torch.utils.data import DataLoader

from data_provider.fusion_window_dataset import Dataset_FusionTimeSeries


data_dict = {
    "FUSION_TS": Dataset_FusionTimeSeries,
}


def data_provider(args, flag):
    if args.data not in data_dict:
        raise ValueError(f"Unsupported custom dataset: {args.data}")

    Data = data_dict[args.data]
    if flag == "test":
        shuffle_flag = False
        drop_last = args.drop_last
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = args.drop_last
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
