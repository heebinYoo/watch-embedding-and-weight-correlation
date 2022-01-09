import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms

"""
거지같은 isc 정리

train - train
test - query
 - gallery

훈련용 클래스의 경우 : 한 클래스당 5개
테스트용 클래스의 경우 : 한 클래스당 쿼리 2개쯤, 겔러리 3개쯤

isc의 평가 : 쿼리 벡터와 겔러리 벡터의 내적 유사도를 만든다음에, 쿼리로 겔러리가 찾아지는지 검증하는 식으로 사용함.

"""


def recall(test_feature_vectors, test_feature_labels, rank):
    test_feature_labels = torch.tensor(test_feature_labels, device=test_feature_vectors.device)

    # 행렬곱으로, 각 벡터간의 내적, 내적 유사도임
    # 원래 논문 구현체에서는 마지막 레이어에 노말라이제이션을 했기 때문에, 단순 행렬곱만으로 코사인 유사도가 나왔었음
    # contiguous t연산이 대상 데이터와 같은 데이터를 포인팅 하는 레퍼런스를 리턴하므로 그걸 메모리 연속이 되게 재할당시키는 애
    sim_matrix = torch.mm(test_feature_vectors, test_feature_labels.t().contiguous())
    # 대각선에 0을 채운다, 자기는 자기랑 제일 닮아서 크게 나올테니까, 근데 자기는 점수에 넣을 생각 없으므로
    sim_matrix.fill_diagonal_(0)

    _, idx = torch.topk(input=sim_matrix, k=rank[-1], dim=-1, largest=True)
    acc_list = []
    for r in rank:
        correct = (test_feature_labels[idx[:, 0:r]] == test_feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / len(test_feature_labels)).item())
    return acc_list


def recall_isc(query_vectors, query_labels, rank, gallery_vectors, gallery_labels):
    query_labels = torch.tensor(query_labels, device=query_vectors.device)
    gallery_labels = torch.tensor(gallery_labels, device=query_vectors.device)

    # 행렬곱으로, 각 벡터간의 내적, 내적 유사도임
    # 원래 논문 구현체에서는 마지막 레이어에 노말라이제이션을 했기 때문에, 단순 행렬곱만으로 코사인 유사도가 나왔었음
    # contiguous t연산이 대상 데이터와 같은 데이터를 포인팅 하는 레퍼런스를 리턴하므로 그걸 메모리 연속이 되게 재할당시키는 애
    sim_matrix = torch.mm(query_vectors, gallery_vectors.t().contiguous())

    _, idx = torch.topk(input=sim_matrix, k=rank[-1], dim=-1, largest=True)
    acc_list = []
    for r in rank:
        correct = (gallery_labels[idx[:, 0:r]] == query_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / len(query_labels)).item())
    return acc_list


class ImageReader(Dataset):

    def __init__(self, data_path, data_name, data_type, crop_type):

        if crop_type == 'cropped' and data_name not in ['cars196', 'CUB_200_2011']:
            raise NotImplementedError('cropped data only works for car or cub dataset')

        data_dict = torch.load('{}/{}/{}_data_dicts.pth'.format(data_path, data_name, crop_type))[data_type]
        self.class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if data_type == 'train':
            self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(224),
                                                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                                                 transforms.ToTensor(), normalize])
        self.images, self.labels = [], []
        for label, image_list in data_dict.items():
            self.images += image_list
            self.labels += [self.class_to_idx[label]] * len(image_list)

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)


class MPerClassSampler(Sampler):
    def __init__(self, labels, batch_size, m=4):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(labels)
        self.batch_size = batch_size
        self.m = m
        assert batch_size % m == 0, 'batch size must be divided by m'

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __iter__(self):
        for _ in range(self.__len__()):
            labels_in_batch = set()
            inds = np.array([], dtype=np.int)

            while inds.shape[0] < self.batch_size:
                sample_label = np.random.choice(self.labels_unique)
                if sample_label in labels_in_batch:
                    continue

                labels_in_batch.add(sample_label)
                sample_label_ids = np.argwhere(np.in1d(self.labels, sample_label)).reshape(-1)
                subsample = np.random.permutation(sample_label_ids)[:self.m]
                inds = np.append(inds, subsample)

            inds = inds[:self.batch_size]
            inds = np.random.permutation(inds)
            yield list(inds)
