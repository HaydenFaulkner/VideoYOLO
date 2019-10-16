"""
Utilised to extract the important information from the logs and output as tab seperated for input into spreadsheet
for graphing
"""


def extract_stats(log_file_path):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]

    stats = dict()
    epoch = 0
    for line in lines:
        if 'Training cost:' in line:
            results = dict()
            epoch = int(line.split(']')[0][7:])
            losses = line.split(',')[1:]
            for loss in losses:
                results[loss.split('=')[0][1:]] = float(loss.split('=')[1])
        elif 'mAP' in line:
            results['mAP'] = float(line.split('=')[1])
            stats[epoch] = results

    return stats


def display_stats(stats, columns=['Epoch', 'mAP', 'ObjLoss', 'BoxCenterLoss', 'BoxScaleLoss', 'ClassLoss'], header=True):

    str_ = ''

    if header:
        for c in columns:
            str_ += c + '\t'
        str_ += '\n'

    for epoch in sorted(stats.keys()):
        for c in columns:
            if c == 'Epoch':
                str_ += str(epoch) + '\t'
            else:
                str_ += str(stats[epoch][c]) + '\t'
        str_ += '\n'

    return str_


if __name__ == '__main__':
    model = '0030'
    stats = extract_stats("models/"+model+"/yolo3_darknet53_vid_train.log")
    print(display_stats(stats, columns=['mAP']))
