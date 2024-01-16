import torch
import torchvision
import torchvision.transforms as transforms
import clip
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torchvision.transforms import ToPILImage
def load_cifar10():
    # 不进行标准化，因为CLIP的预处理会处理这部分
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    return trainset, testset

def split_data(trainset, num_clients, alpha):
    num_items = len(trainset)
    indices = np.random.dirichlet(np.repeat(alpha, num_clients), num_items)
    indices = (num_items * indices).cumsum().astype(int)
    # 调整最后一个索引以确保所有数据被包含
    indices[-1] = num_items
    indices = np.split(np.random.permutation(num_items), indices[:-1])
    return [Subset(trainset, idx) for idx in indices]



def get_features_from_clip(dataloader, model, preprocess, device):
    features = []
    to_pil = ToPILImage()  # 创建将张量转换为PIL.Image的转换器
    for images, _ in dataloader:
        processed_images = [preprocess(to_pil(image.to('cpu'))) for image in images]  # 将张量转换为PIL.Image
        images = torch.stack(processed_images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
        features.append(image_features.cpu().numpy())
    return np.concatenate(features, axis=0)



def train_client_model(client_data, clip_model, preprocess, device):
    client_loader = DataLoader(client_data, batch_size=32, shuffle=True)
    features = get_features_from_clip(client_loader, clip_model, preprocess, device)
    labels = np.array([label for _, label in client_loader.dataset])

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(features, labels)
    return logistic_model


def aggregate_models(models):
    avg_weights = np.mean([model.coef_ for model in models], axis=0)
    avg_bias = np.mean([model.intercept_ for model in models], axis=0)
    aggregated_model = LogisticRegression()
    aggregated_model.coef_ = avg_weights
    aggregated_model.intercept_ = avg_bias
    return aggregated_model


def evaluate_model(model, dataloader, device, clip_model, preprocess):
    model.to(device)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = torch.stack([preprocess(image) for image in images]).to(device)
            features = clip_model.encode_image(images).cpu().numpy()
            preds = model.predict(features)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def main():
    # 设定参数
    num_clients = 10
    alpha = 0.5

    # 加载数据集和模型
    trainset, testset = load_cifar10()
    client_datasets = split_data(trainset, num_clients, alpha)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 在客户端上训练模型
    client_models = []
    for client_data in client_datasets:
        model = train_client_model(client_data, clip_model, preprocess, device)
        client_models.append(model)

    # 在服务器端进行聚合
    aggregated_model = aggregate_models(client_models)

    # 评估聚合后的模型
    accuracy = evaluate_model(aggregated_model, test_loader, device, clip_model, preprocess)
    print(f"Aggregated Model Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
