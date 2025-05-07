import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import random_split, DataLoader
from model.utils import data_utils

ROOT_DATA_DIR = './data/'

class Pab1():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='pab1',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_data_file = '{}-train.csv'.format(dataset)
        self.test_data_file = '{}-test.csv'.format(dataset)

        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'pab1/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'pab1/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_pab1_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_pab1_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_labels, v_labels = train_test_split(*train_all_data_numpy,
                                                                                        train_size=train_size,
                                                                                        random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_labels]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_labels]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class bgl3():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='bgl3',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_data_file = '{}-train.csv'.format(dataset)
        self.test_data_file = '{}-test.csv'.format(dataset)


        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'bgl3/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'bgl3/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_bgl3_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_bgl3_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class gifford():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='gifford',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_data_file = 'Gifford-train.csv'
        self.test_data_file = 'Gifford-test.csv'


        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'gifford_data/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'gifford_data/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_giff_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_giff_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class NESP():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='NESP',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_data_file = '{}-train.csv'.format(dataset)
        self.test_data_file = '{}-test.csv'.format(dataset)


        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'NESP/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'NESP/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_nesp_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_nesp_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class ube4b():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='ube4b',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size

        self.train_data_file = '{}-train.csv'.format(dataset)
        self.test_data_file = '{}-test.csv'.format(dataset)


        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'ube4b/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'ube4b/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_ube4b_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_ube4b_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class HIS7():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='HIS7',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_data_file = '{}-train.csv'.format(dataset)
        self.test_data_file = '{}-test.csv'.format(dataset)


        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'indels/HIS7/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'indels/HIS7/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_HIS7_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_HIS7_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class CAPSD():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='CAPSD',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size

        self.train_data_file = '{}-train.csv'.format(dataset)
        self.test_data_file = '{}-test.csv'.format(dataset)


        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'indels/CAPSD/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'indels/CAPSD/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_CAPSD_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_CAPSD_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class B1LPA6():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='B1LPA6',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size

        self.train_data_file = '{}-train.csv'.format(dataset)
        self.test_data_file = '{}-test.csv'.format(dataset)


        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'indels/B1LPA6/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'indels/B1LPA6/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_B1LPA6_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_B1LPA6_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class GFP():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='GFP',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size

        self.train_data_file = '{}-train.csv'.format(dataset)
        self.test_data_file = '{}-test.csv'.format(dataset)


        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'mut_data/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'mut_data/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_GFP_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_GFP_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]

        # reconstruction
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class TAPE():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='TAPE',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size

        self.train_data_file = '{}-train.csv'.format(dataset)
        self.test_data_file = '{}-test.csv'.format(dataset)


        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'mut_data/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'mut_data/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_TAPE_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_TAPE_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]

        # reconstruction
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class MSA():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='MSA',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size

        self.train_data_file = 'luxafilt_llmsa_train.csv'
        self.test_data_file = 'luxafilt_llmsa_test.csv'

        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'MSA/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'MSA/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_MSA_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_MSA_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class MSA_RAW():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='MSA_RAW',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size

        self.train_data_file = 'll_train.csv'
        self.test_data_file = 'll_test.csv'

        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'MSA/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'MSA/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_MSA_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_MSA_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


class MDH():
    def __init__(self, data_dir=ROOT_DATA_DIR,
                 dataset='MDH',
                 batch_size=100):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size

        self.train_data_file = 'train_sequences.csv'
        self.test_data_file = 'val_sequences.csv'

        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'MDH/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + 'MDH/' + self.test_data_file))


        train_seqs, train_fitness, train_labels = data_utils.load_raw_MDH_data(train_df)
        test_seqs, test_fitness, test_labels = data_utils.load_raw_MDH_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness, train_labels)
        self.raw_test_tup = (test_seqs, test_fitness, test_labels)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        # reconstruction
        train_data = self.raw_train_tup[0]
        train_targets = self.raw_train_tup[0]
        test_data = self.raw_test_tup[0]
        test_targets = self.raw_test_tup[0]

        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_labels = self.raw_train_tup[2]
        test_labels = self.raw_test_tup[2]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, train_labels]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_label, v_label = train_test_split(*train_all_data_numpy,
                                                                                      train_size=train_size,
                                                                                      random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_label]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_label]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness, test_labels)

    def setup(self):

        self.train_split, self.valid_split = self.train_dataset, self.valid_dataset

        print("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)


def str2data(dataset_name):

    if dataset_name == 'gifford':
        data = gifford

    elif dataset_name == 'GFP':
        data = GFP

    elif dataset_name == 'TAPE':
        data = TAPE

    elif dataset_name == 'pab1':
        data = Pab1

    elif dataset_name == 'bgl3':
        data = bgl3

    elif dataset_name == 'NESP':
        data = NESP

    elif dataset_name == 'ube4b':
        data = ube4b

    elif dataset_name == 'HIS7':
        data = HIS7

    elif dataset_name == 'CAPSD':
        data = CAPSD

    elif dataset_name == 'B1LPA6':
        data = B1LPA6

    elif dataset_name == 'MSA':
        data = MSA

    elif dataset_name == 'MSA_RAW':
        data = MSA_RAW

    elif dataset_name == 'MDH':
        data = MDH

    else:
        raise NotImplementedError(f'{dataset_name} not implemented')

    return data

