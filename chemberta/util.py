import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem

def pic50_to_ic50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)


def reorder_canonical_rank_atoms(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order


def smiles2graph(smiles_string, remove_hs=True, reorder_atoms=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    mol = Chem.MolFromSmiles(smiles_string)
    mol = mol if remove_hs else Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = reorder_canonical_rank_atoms(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_attr'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph


def huber_loss(y_pred, y_true, delta=0.5):
    # Calculate the absolute error
    abs_error = torch.abs(y_true - y_pred)

    # Calculate the Huber loss based on the absolute error
    loss = torch.where(abs_error <= delta,
                       1 * abs_error ** 2,
                       delta * (abs_error - 0.5 * delta))

    # Return the mean loss
    return torch.mean(loss)


# 0.5 = B 값의 threshold는 0.5 임
def mse_threshold_no_learn(y_true, y_pred, epsilon=0.5):
    error = torch.abs(y_pred - y_true)
    loss = torch.where(error < epsilon, 0, (y_pred - y_true) ** 2)
    return torch.mean(loss)


class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Ensure no negative values by clipping to 0
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        # Logarithmic transformation (log(1 + x))
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)

        # Compute MSE between the logarithms
        return self.mse(log_pred, log_true)

def get_name(obj):
    # 함수인지 확인
    if callable(obj):
        return obj.__name__
    # 객체인 경우 클래스 이름 반환
    else:
        return obj.__class__.__name__


class LossCalculator(nn.Module):
    def __init__(self, criterion=nn.MSELoss()):
        super(LossCalculator, self).__init__()

        self.criterion = criterion

        print('using loss:', type(criterion))

    def epoch(self, epoch=None):
        self.current = epoch
        self.losses = []
        self.errors = np.zeros(0)
        self.mses = np.zeros(0)
        self.predictions = np.zeros(0)

    def print_status(self, validation=False):
        A = np.mean(np.sqrt(self.mses) / (np.max(self.predictions) - np.min(self.predictions))).item()
        A = (1 - min(A, 1))

        B = np.mean(self.errors <= 0.5).item()

        score = 0.5 * A + 0.5 * B
        loss = sum(self.losses) / max(len(self.losses), 1)

        print('=== epoch {}{}==='.format(self.current, ' (Validation) ' if validation else ' '))
        print('loss: {:.8f} score: {:.8f}, A: {:.3f} B: {:.3f}'.format(loss, score, A, B))

    def forward(self, prediction, target):
        loss = self.criterion(prediction, target)

        prediction = prediction.cpu().detach().numpy().squeeze()
        target = target.cpu().detach().numpy().squeeze()

        if prediction.ndim == 0:
            prediction = np.expand_dims(prediction, axis=0)

        if target.ndim == 0:
            target = np.expand_dims(target, axis=0)

        loss_all = np.abs(prediction - target) ** 2
        abs_error_ic50 = np.abs(pic50_to_ic50(prediction) - pic50_to_ic50(target))

        self.losses.append(loss.item())
        self.errors = np.concatenate((self.errors, abs_error_ic50))
        self.mses = np.concatenate((self.mses, loss_all))
        self.predictions = np.concatenate((self.predictions, prediction))

        return loss


class ThresholdPenaltyLoss(nn.Module):
    def __init__(self, threshold, penalty_weight):
        super(ThresholdPenaltyLoss, self).__init__()
        self.threshold = threshold  # 임계값
        self.penalty_weight = penalty_weight  # 벌점 가중치
        self.mse = nn.MSELoss()  # 기본 손실 함수 (MSE 사용)

    def forward(self, predictions, targets):
        # 기본 MSE 손실 계산
        mse_loss = self.mse(predictions, targets)

        # 임계값을 넘는 예측에 대해 벌점 부과
        over_threshold = torch.relu(predictions - self.threshold)  # 임계값을 넘는 부분
        penalty = self.penalty_weight * torch.sum(over_threshold)  # 넘는 부분에 대해 벌점 부과

        # 최종 손실 = MSE + 벌점
        total_loss = mse_loss + penalty

        return total_loss


def zero_index(value):
    return value[:, 0, ...]


def get_statistical(value, axis=1, only_mean=True):
    mean_val = torch.mean(value, dim=axis)

    feature_list = [mean_val]
    if not only_mean:
        std_val = torch.std(value, dim=axis)
        median_val = torch.median(value, dim=axis)
        max_val = torch.max(value, dim=axis)
        min_val = torch.min(value, dim=axis)
        feature_list.extend([std_val, median_val, max_val, min_val])

    concatenated = np.concatenate(feature_list)
    return concatenated


def make_context_feature(feature, context_window_size=5):
    # feature: [batch_size, sequence_length, feature_dim]
    batch_size, sequence_length, feature_dim = feature.shape

    # 패딩 적용 (배치, 시퀀스, 특징)
    # padding을 앞뒤로 context_window_size만큼 적용
    padded_feature = F.pad(feature, (0, 0, context_window_size, context_window_size), "constant", 0)

    # 새로운 특징 벡터를 저장할 공간 할당
    contextual_features = []

    # 각 타임스텝에서 컨텍스트를 가져와서 벡터화
    for i in range(context_window_size, sequence_length + context_window_size):
        # 문맥 윈도우 내의 데이터 추출
        context = padded_feature[:, i - context_window_size:i + context_window_size + 1, :].reshape(batch_size, -1)
        contextual_features.append(context)

    # 리스트를 텐서로 변환하고 쌓기
    contextual_features = torch.stack(contextual_features, dim=1)

    return contextual_features