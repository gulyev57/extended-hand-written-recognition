# Gerekli kütüphaneler içe aktarılıyor
# PyTorch, görsel işleme için Pillow, arayüz için Tkinter ve numpy dahil ediliyor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
from scipy.ndimage import center_of_mass, shift

# GPU varsa kullan, yoksa CPU ile devam et
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Konvolüsyonel sinir ağı tanımı (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 kanal (gri tonlama) giriş, 32 filtreli ilk konvolüsyon katmanı
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 32 kanal giriş, 64 filtreli ikinci konvolüsyon katmanı
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        # 9216 giriş, 128 çıkışlı tam bağlantılı katman
        self.fc1 = nn.Linear(9216, 128)
        # 128 giriş, 10 çıkış (0-9 rakamları)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Katmanlar sırasıyla uygulanıyor
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Görüntüyü merkezlemek için ağırlık merkezine göre kaydırma fonksiyonu
def center_image(img_np):
    cy, cx = center_of_mass(img_np)
    shiftx = int(round(img_np.shape[1]/2.0 - cx))
    shifty = int(round(img_np.shape[0]/2.0 - cy))
    return shift(img_np, shift=[shifty, shiftx], mode='constant', cval=0.0)

# EMNIST modeli için kullanıcı çizimini uygun formata dönüştürme
def process_tkinter_image(image):
    image = image.resize((280, 280)).convert("L")  # Gri tonlama
    image = ImageOps.invert(image)  # Siyah-beyazı ters çevir
    image = image.transpose(Image.TRANSPOSE)  # Diyagonal yansıma
    image = ImageEnhance.Contrast(image).enhance(2.0)  # Kontrast artır
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    else:
        image = Image.new("L", (28, 28), 0)
    max_side = max(image.size)
    scale = 20 / max_side
    new_size = (int(image.size[0]*scale), int(image.size[1]*scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new("L", (28, 28), 0)
    paste_pos = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    new_img.paste(image, paste_pos)
    img_np = np.array(new_img).astype(np.float32) / 255.0
    img_np = center_image(img_np)
    img_np = (img_np - 0.1307) / 0.3081  # Normalize
    return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

# MNIST modeli için kullanıcı çizimini uygun formata dönüştürme
def process_tkinter_image_for_mnist(image):
    image = image.resize((280, 280)).convert("L")
    image = ImageOps.invert(image)
    image = ImageEnhance.Contrast(image).enhance(2.0)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    else:
        image = Image.new("L", (28, 28), 0)
    max_side = max(image.size)
    scale = 20 / max_side
    new_size = (int(image.size[0]*scale), int(image.size[1]*scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new("L", (28, 28), 0)
    paste_pos = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    new_img.paste(image, paste_pos)
    img_np = np.array(new_img).astype(np.float32) / 255.0
    img_np = center_image(img_np)
    img_np = (img_np - 0.1307) / 0.3081
    return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

# Tkinter arayüzü başlatma fonksiyonu
def run_tkinter_app(emnist_model, mnist_model):
    window = tk.Tk()
    window.title("Digit Recognizer - EMNIST vs MNIST")
    canvas_width, canvas_height = 280, 280

    # Çizim için canvas oluştur
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack()

    # Canvas üzerinde çizim yapılacak bir PIL nesnesi
    image1 = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image1)

    # EMNIST ve MNIST sonuçlarını gösterecek StringVar değişkenleri
    result_var = tk.StringVar()
    mnist_result_var = tk.StringVar()

    # EMNIST sonucu etiketi
    label = tk.Label(window, textvariable=result_var, font=("Arial", 18))
    label.pack()

    # MNIST sonucu etiketi (mavi renkle)
    mnist_label = tk.Label(window, textvariable=mnist_result_var, font=("Arial", 18), fg="blue")
    mnist_label.pack()

    # Kullanıcının fareyle çizim yapmasını sağlar
    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        draw.ellipse([x1, y1, x2, y2], fill="black")

    # Tahmin yapar: her iki model için ayrı ayrı
    def predict():
        img_tensor = process_tkinter_image(image1)
        img_tensor_mnist = process_tkinter_image_for_mnist(image1)
        with torch.no_grad():
            # EMNIST tahmini
            output_e = emnist_model(img_tensor)
            probs_e = F.softmax(output_e, dim=1)[0]
            pred_e = output_e.argmax(dim=1).item()
            conf_e = probs_e[pred_e].item() * 100
            result_var.set(f"EMNIST Prediction: {pred_e} ({conf_e:.2f}%)")

            # MNIST tahmini
            output_m = mnist_model(img_tensor_mnist)
            probs_m = F.softmax(output_m, dim=1)[0]
            pred_m = output_m.argmax(dim=1).item()
            conf_m = probs_m[pred_m].item() * 100
            mnist_result_var.set(f"MNIST Prediction: {pred_m} ({conf_m:.2f}%)")

    # Canvas’ı temizler
    def clear():
        canvas.delete("all")
        draw.rectangle([0, 0, canvas_width, canvas_height], fill="white")
        result_var.set("")
        mnist_result_var.set("")

    # Uygulamayı kapatır
    def quit_app():
        window.destroy()

    # Fare hareketini çizim ile ilişkilendir
    canvas.bind("<B1-Motion>", paint)

    # Butonlar
    btn_frame = tk.Frame(window)
    btn_frame.pack()
    tk.Button(btn_frame, text="Predict", command=predict).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(btn_frame, text="Clear", command=clear).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(btn_frame, text="Quit", command=quit_app).pack(side=tk.LEFT, padx=5, pady=5)

    window.mainloop()

# MNIST modelini eğitir ve kaydeder
def train_and_save_mnist_model():
    model = CNN().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000, shuffle=False)

    for epoch in range(1, 5):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Eğitilen model dosyaya kaydedilir
    torch.save(model.state_dict(), "mnist_model.pt")
    print("MNIST model saved.")

# Ana çalışma kısmı
if __name__ == "__main__":
    # EMNIST modelini yükle
    emnist_model = CNN().to(device)
    emnist_model.load_state_dict(torch.load("emnist_digits_cnn_v2.pt", map_location=device))
    emnist_model.eval()

    # MNIST modeli yoksa eğit ve kaydet
    if not os.path.exists("mnist_model.pt"):
        train_and_save_mnist_model()

    # MNIST modelini yükle
    mnist_model = CNN().to(device)
    mnist_model.load_state_dict(torch.load("mnist_model.pt", map_location=device))
    mnist_model.eval()

    # Uygulamayı başlat
    run_tkinter_app(emnist_model, mnist_model)
