import torch.nn.functional as F


def seq_loss(inp, lens, labels):
    total_loss = 0
    for i in range(len(inp)):
        total_loss += F.cross_entropy(inp[i, :lens[i], :], labels[i][:lens[i]], reduction='sum')
    return total_loss / sum(lens)


def calc_accuracy(inp_inds, output, true_labels):
    mask = (inp_inds != 0)
    total_preds = mask.sum()
    correct = ((output.argmax(dim=2) == true_labels) * mask).sum()
    return correct.float() / total_preds.item()