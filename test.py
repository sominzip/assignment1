import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. Data Loader
# =========================
def load_data(dataset="mnist", batch_size=64):
    transform = transforms.ToTensor()

    if dataset == "mnist":
        trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    elif dataset == "cifar":
        trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# =========================
# 2. Model
# =========================
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7,128), nn.ReLU(), nn.Linear(128,10)
        )

    def forward(self,x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0),-1))


class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*4*4,256), nn.ReLU(), nn.Linear(256,10)
        )

    def forward(self,x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0),-1))

# =========================
# 3. Train / Eval
# =========================
def train(model, loader, epochs=5):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for e in range(epochs):
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        print(f"Epoch {e+1} done")

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
    acc = correct / len(loader.dataset)
    print("Clean Acc:", acc)
    return acc

# =========================
# 4. FGSM / PGD
# =========================
def fgsm_untargeted(model, x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    model.zero_grad()
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    return torch.clamp(x_adv,0,1).detach()

def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), target)
    model.zero_grad()
    loss.backward()
    x_adv = x_adv - eps * x_adv.grad.sign()
    return torch.clamp(x_adv,0,1).detach()

def pgd_untargeted(model, x, y, eps=0.3, step=0.01, k=10):
    x_adv = x.clone().detach()
    for _ in range(k):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv + step * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+eps), x-eps)
        x_adv = torch.clamp(x_adv,0,1).detach()
    return x_adv

def pgd_targeted(model, x, target, eps=0.3, step=0.01, k=10):
    x_adv = x.clone().detach()
    for _ in range(k):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), target)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv - step * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+eps), x-eps)
        x_adv = torch.clamp(x_adv,0,1).detach()
    return x_adv

# =========================
# 5. Attack Success Rate
# =========================
def attack_success_rate(model, loader, attack_fn, targeted=False):
    model.eval()
    success, total = 0,0

    for x,y in loader:
        x,y = x.to(device), y.to(device)

        if targeted:
            target = (y+1)%10
            x_adv = attack_fn(model,x,target)
        else:
            x_adv = attack_fn(model,x,y)

        with torch.no_grad():
            pred = model(x_adv).argmax(1)

        success += ((pred==target) if targeted else (pred!=y)).sum().item()

        total += y.size(0)
        if total >= 100: break

    return success/total

# =========================
# 6. Save Images
# =========================
def save_images(model, loader, attack_fn, name, dataset):
    os.makedirs("results", exist_ok=True)
    model.eval()
    saved = 0

    for x,y in loader:
        x,y = x.to(device), y.to(device)
        x_adv = attack_fn(model,x,y)

        with torch.no_grad():
            p1 = model(x).argmax(1)
            p2 = model(x_adv).argmax(1)

        for i in range(x.size(0)):
            if saved >= 5: return

            if dataset == "mnist":
                orig = x[i].cpu().squeeze().numpy()
                adv = x_adv[i].cpu().squeeze().numpy()
                diff = (adv - orig)*10
                cmap='gray'
            else:
                orig = x[i].cpu().permute(1,2,0).numpy()
                adv = x_adv[i].cpu().permute(1,2,0).numpy()
                diff = (adv - orig)*5
                cmap=None

            plt.figure(figsize=(9,3))
            for j, img in enumerate([orig, adv, diff]):
                plt.subplot(1,3,j+1)
                plt.imshow(img, cmap=cmap)
                plt.axis('off')

            plt.savefig(f"results/{name}_{dataset}_{saved}.png")
            plt.close()
            saved += 1

# =========================
# 7. Run One Dataset
# =========================

def run(dataset):
    print(f"\n===== {dataset.upper()} =====")

    train_loader, test_loader = load_data(dataset)

    model = MNIST_CNN().to(device) if dataset=="mnist" else CIFAR_CNN().to(device)

    train(model, train_loader)
    evaluate(model, test_loader)

    # =========================
    # setting parameters
    # =========================
    if dataset == "mnist":
        fgsm_eps = 0.2
        pgd_eps = 0.2
        pgd_step = 0.01
        pgd_k = 20
    else:  # CIFAR
        fgsm_eps = 0.02
        pgd_eps = 0.02
        pgd_step = 0.003
        pgd_k = 7

    # =========================
    # FGSM
    # =========================
    print("FGSM Untargeted:",
          attack_success_rate(model,test_loader,
            lambda m,x,y: fgsm_untargeted(m,x,y,fgsm_eps)))

    print("FGSM Targeted:",
          attack_success_rate(model,test_loader,
            lambda m,x,y: fgsm_targeted(m,x,y,fgsm_eps), True))

    # =========================
    # PGD
    # =========================
    print("PGD Untargeted:",
          attack_success_rate(model,test_loader,
            lambda m,x,y: pgd_untargeted(m,x,y,
                                        eps=pgd_eps,
                                        step=pgd_step,
                                        k=pgd_k)))

    print("PGD Targeted:",
          attack_success_rate(model,test_loader,
            lambda m,x,y: pgd_targeted(m,x,y,
                                      eps=pgd_eps,
                                      step=pgd_step,
                                      k=pgd_k), True))

    # =========================
    # save images
    # =========================
    save_images(model,test_loader,
                lambda m,x,y: fgsm_untargeted(m,x,y,fgsm_eps),
                "fgsm", dataset)

    save_images(model,test_loader,
                lambda m,x,y: pgd_untargeted(m,x,y,
                                            eps=pgd_eps,
                                            step=pgd_step,
                                            k=pgd_k),
                "pgd", dataset)

# =========================
# 8. Main
# =========================
if __name__ == "__main__":
    run("mnist")
    run("cifar")
