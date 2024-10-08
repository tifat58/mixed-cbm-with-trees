import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import KBinsDiscretizer
from torch.nn.functional import one_hot


def load_vDem(base_dir='./data'):
    data = pd.read_csv(f'./datasets/vDem/V-Dem-CY-Core-v10.csv')
    data['country_name_year'] = data['country_name'] + '_' + data['year'].astype(str)
    data_2000 = data[data['year'] > 2000].iloc[:, 12:-1].dropna(axis=1)

    high_level_indicators = [
        'v2x_polyarchy',
        # 'v2x_libdem',
        # 'v2x_partipdem',
        'v2x_delibdem',
        'v2x_egaldem'
    ]
    mid_level_indicators = [
        'v2x_api',
        'v2x_mpi',
        'v2x_freexp_altinf',
        'v2x_frassoc_thick',
        'v2x_suffr',
        'v2xel_frefair',
        'v2x_elecoff',
        # 'v2x_liberal',
        'v2xcl_rol',
        # 'v2x_jucon',
        # 'v2xlg_legcon',
        # 'v2x_partip',
        'v2x_cspart',
        # 'v2xdd_dd',
        # 'v2xel_locelec',
        # 'v2xel_regelec',
        'v2xdl_delib',
        'v2x_egal',
        'v2xeg_eqprotec',
        'v2xeg_eqaccess',
        'v2xeg_eqdr',
    ]

    # drop_list = ['codelow', 'codehigh', 'sd', 'osp', 'nr', 'mean']
    low_level_indicators = []
    for f in data_2000.columns:
        if f.endswith('_ord') and f not in high_level_indicators and f not in mid_level_indicators:
            low_level_indicators.append(f)

    low_level_indicators_continuous = []
    for f in data_2000.columns:
        if f.endswith('_codehigh') or f.endswith('_codelow') and \
                f not in high_level_indicators and f not in mid_level_indicators:
            low_level_indicators_continuous.append(f)

    print(f'Main {len(high_level_indicators)} - Area {len(mid_level_indicators)} - Raw {len(low_level_indicators)}')

    data_low_continuous = data_2000[low_level_indicators_continuous]

    data_low_raw = data_2000[low_level_indicators]
    one_hots = []
    for indicator in low_level_indicators:
        c = data_low_raw[indicator].values
        n_bins = int(c.max())
        kbin = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='uniform')
        c1h = kbin.fit_transform(c.reshape(-1, 1))
        one_hots.append(c1h)

    new_indicator_names = []
    for clist, cname in zip(one_hots, low_level_indicators):
        if clist.shape[1] > 1:
            for i in range(clist.shape[1]):
                new_indicator_names.append(f'{cname}_{i}')
        else:
            new_indicator_names.append(f'{cname}')

    data_low = pd.DataFrame(np.hstack(one_hots), columns=new_indicator_names)
    data_mid = data_2000[mid_level_indicators] > 0.5
    data_high = data_2000[high_level_indicators].iloc[:, 0] > 0.5

    # data_mid = pd.DataFrame(np.hstack([data_low, data_mid]), columns=data_low.columns.append(data_mid.columns))
    #
    # scores = cross_val_score(LogisticRegression(), data_mid.values, data_high.values, cv=10)
    # print(scores.mean())
    # scores = cross_val_score(DecisionTreeClassifier(), data_mid.values, data_high.values, cv=10)
    # print(scores.mean())
    #
    # # visualise the tree
    # from sklearn.tree import export_graphviz
    # import pydotplus
    # from IPython.display import Image
    # model = DecisionTreeClassifier(max_depth=100)
    # model.fit(data_mid.values, data_high.values)
    # dot_data = export_graphviz(model, out_file=None, feature_names=data_mid.columns)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())
    # graph.write_png('./tree_msl=45.png')
    #
    # scores = cross_val_score(DecisionTreeClassifier(min_samples_leaf=100), data_mid.values, data_high.values, cv=10)
    # print(scores.mean())

    x = torch.FloatTensor(data_low.values)
    c = torch.FloatTensor(data_mid.values)
    y = one_hot(torch.tensor(data_high.values).to(torch.long)).to(torch.float)
    return x, c, y, data_mid.columns


if __name__ == '__main__':
    x, c, y, feature_names = load_vDem()