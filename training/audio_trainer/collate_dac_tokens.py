from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):

    x_list = []
    y_list = []

    for x, y in batch:
        x_list.append(x)
        y_list.append(y)

    x = pad_sequence(
        x_list,
        batch_first=True,
        padding_value=0
    )

    y = pad_sequence(
        y_list,
        batch_first=True,
        padding_value=-100
    )

    return x, y