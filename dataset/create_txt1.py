import pandas as pd


def create_txt(ratio=0.9):
    df = pd.read_csv('train.csv')  # 读取数据
    val = pd.DataFrame()  # 划分出的test集合
    train = pd.DataFrame()  # 剩余的train集合
    tags = df['label'].unique().tolist()  # 按照该标签进行等比例抽取

    for tag in tags:
        # 随机选取0.2的数据
        data = df[(df['label'] == tag)]
        sample = data.sample(int((1 - ratio) * len(data)))
        sample_index = sample.index
        # 剩余数据
        all_index = data.index
        residue_index = all_index.difference(sample_index)  # 去除sample之后剩余的数据
        residue = data.loc[residue_index]  # 这里要使用.loc而非.iloc
        # 保存
        val = pd.concat([val, sample], ignore_index=True)
        train = pd.concat([train, residue], ignore_index=True)

    # 保存为 分隔的文本
    val['image_id'] = './dataset/train/' + val['image_id']
    train['image_id'] = './dataset/train/' + train['image_id']
    val.to_csv('val.txt', sep=' ', index=False, header=False)
    train.to_csv('train.txt', sep=' ', index=False, header=False)


if __name__ == '__main__':
    create_txt(0.9)