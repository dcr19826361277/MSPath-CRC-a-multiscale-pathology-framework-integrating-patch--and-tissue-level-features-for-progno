#混淆矩阵
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from torchvision import transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.font_manager as fm

# 创建结果保存目录
result_dir = 'E:\WSI-HSfeature/analysis_results'
os.makedirs(result_dir, exist_ok=True)

# 设置字体为Arial加粗
font_path = None
# 尝试查找Arial字体路径
for font in fm.fontManager.ttflist:
    if 'arial' in font.name.lower():
        font_path = font.fname
        break

if font_path:
    arial_font = fm.FontProperties(fname=font_path, weight='bold')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
else:
    print("未找到Arial字体，将使用默认字体")

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'使用设备: {device}')


# 定义数据集类
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        imagelist = []
        lablelist = []
        # 检查img_dir是否存在
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"目录不存在: {img_dir}")

        # 遍历所有子目录
        for folder in os.listdir(img_dir):
            folder_path = os.path.join(img_dir, folder)
            # 只处理目录
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    # 确保处理的是文件而不是子目录
                    if os.path.isfile(file_path):
                        imagelist.append(file)
                        lablelist.append(folder)

        self.imagelist = imagelist
        self.lablelist = lablelist
        print(f"加载了 {len(imagelist)} 张图像，分为 {len(set(lablelist))} 个类别")

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.lablelist[idx], self.imagelist[idx])
        # 检查文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")

        image = Image.open(img_path)
        labels_map = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}
        label = labels_map[self.lablelist[idx]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载测试数据集
try:
    test_dataset = ImageDataset(r'E:\WSI-HSfeature\CRC-VAL-HE-7K', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试数据集加载成功，共 {len(test_dataset)} 个样本")
except Exception as e:
    print(f"加载测试数据集时出错: {e}")
    # 如果加载失败，可以在这里添加备选方案或者退出程序
    import sys

    sys.exit(1)

# 加载模型
model = models.resnet50(weights=None)
# 修改最后一层以匹配我们的类别数
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 1024),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(1024, 9)
)

try:
    model.load_state_dict(torch.load('E:\WSI-HSfeature/resnet50_weights.pth'))
    print("模型权重加载成功")
except Exception as e:
    print(f"加载模型权重时出错: {e}")
    sys.exit(1)

model = model.to(device)
model.eval()

# 获取模型预测
y_true = []
y_score = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        # 使用softmax获取概率
        probs = torch.nn.functional.softmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_score.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_score = np.array(y_score)

# 将标签二值化
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
n_classes = y_true_bin.shape[1]

# 计算每个类别的ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 类别标签映射
labels_map = {0: "ADI", 1: "BACK", 2: "DEB", 3: "LYM", 4: "MUC", 5: "MUS", 6: "NORM", 7: "STR", 8: "TUM"}

# 绘制每个类别的ROC曲线（单独绘制）
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
for i, color in zip(range(n_classes), colors):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr[i], tpr[i], color=color, lw=3,
             label=f'ROC 曲线 (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率', fontsize=12, fontweight='bold')
    plt.ylabel('真阳性率', fontsize=12, fontweight='bold')
    plt.title(f'{labels_map[i]}类别的ROC曲线', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", prop={'weight': 'bold', 'size': 10})
    plt.grid(True)

    # 保存图像
    plt.savefig(os.path.join(result_dir, f'ROC_{labels_map[i]}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 计算混淆矩阵
y_pred = np.argmax(y_score, axis=1)
cm = confusion_matrix(y_true, y_pred)

# 打印分类报告
class_names = list(labels_map.values())
print("\n分类报告:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 可视化混淆矩阵（调整字号和样式）
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵', fontsize=16, fontweight='bold')
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, fontsize=12, fontweight='bold')
plt.yticks(tick_marks, class_names, fontsize=12, fontweight='bold')

# 在混淆矩阵上标注数值（调整字号）
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 fontsize=12, fontweight='bold',
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('真实标签', fontsize=14, fontweight='bold')
plt.xlabel('预测标签', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"所有结果已保存到 {result_dir} 文件夹中")

#分类ROC
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from torchvision import transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.font_manager as fm

# 创建结果保存目录
result_dir = 'E:\WSI-HSfeature/analysis_results'
os.makedirs(result_dir, exist_ok=True)


# 字体配置类
class FontConfig:
    def __init__(self, font_size=35, font_weight='bold', font_family=None):
        self.font_size = font_size
        self.font_weight = font_weight
        self.font_family = font_family
        self.font_path = None

    def setup_font(self):
        """配置matplotlib字体"""
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['font.weight'] = self.font_weight
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 字体查找逻辑
        if self.font_family:
            plt.rcParams['font.family'] = self.font_family
            return True

        try:
            # 优先查找Arial字体
            for font in fm.fontManager.ttflist:
                if 'arial' in font.name.lower():
                    self.font_path = font.fname
                    break
            # 若未找到Arial，尝试查找其他常用无衬线字体
            if not self.font_path:
                for font in fm.fontManager.ttflist:
                    if 'sans-serif' in font.style.lower() or 'helvetica' in font.name.lower():
                        self.font_path = font.fname
                        break
        except Exception as e:
            print(f"字体查找出错: {e}")
            return False

        if self.font_path:
            plt.rcParams['font.family'] = 'Arial'
            print(f"已设置字体: Arial, 大小: {self.font_size}, 粗细: {self.font_weight}")
            return True
        else:
            print(f"未找到指定字体，将使用默认字体，大小: {self.font_size}, 粗细: {self.font_weight}")
            return False


# 定义设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'使用设备: {device}')


# 定义数据集类
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        imagelist = []
        lablelist = []
        # 检查img_dir是否存在
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"目录不存在: {img_dir}")

        # 遍历所有子目录
        for folder in os.listdir(img_dir):
            folder_path = os.path.join(img_dir, folder)
            # 只处理目录
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    # 确保处理的是文件而不是子目录
                    if os.path.isfile(file_path):
                        imagelist.append(file)
                        lablelist.append(folder)

        self.imagelist = imagelist
        self.lablelist = lablelist
        print(f"加载了 {len(imagelist)} 张图像，分为 {len(set(lablelist))} 个类别")

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.lablelist[idx], self.imagelist[idx])
        # 检查文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")

        image = Image.open(img_path)
        labels_map = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}
        label = labels_map[self.lablelist[idx]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 评估类
class ModelEvaluator:
    def __init__(self, model, test_dataloader, result_dir, font_config=None):
        self.model = model
        self.test_dataloader = test_dataloader
        self.result_dir = result_dir
        self.font_config = font_config or FontConfig()
        self.labels_map = {0: "ADI", 1: "BACK", 2: "DEB", 3: "LYM", 4: "MUC", 5: "MUS", 6: "NORM", 7: "STR", 8: "TUM"}
        self.y_true = None
        self.y_score = None

    def evaluate(self, save_predictions=True):
        """执行模型评估并保存结果"""
        # 设置字体
        self.font_config.setup_font()

        # 获取模型预测
        self.y_true = []
        self.y_score = []

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                # 使用softmax获取概率
                probs = torch.nn.functional.softmax(outputs, dim=1)
                self.y_true.extend(labels.cpu().numpy())
                self.y_score.extend(probs.cpu().numpy())

        self.y_true = np.array(self.y_true)
        self.y_score = np.array(self.y_score)

        # 保存预测结果
        if save_predictions:
            np.save(os.path.join(self.result_dir, 'y_true.npy'), self.y_true)
            np.save(os.path.join(self.result_dir, 'y_score.npy'), self.y_score)
            print(f"预测结果已保存到 {self.result_dir}")

        return self.y_true, self.y_score

    def plot_roc_curves(self):
        """绘制ROC曲线（无标题，仅保留坐标轴标签，AUC标签放大）"""
        if self.y_true is None or self.y_score is None:
            print("请先执行评估")
            return

        # 将标签二值化
        y_true_bin = label_binarize(self.y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        n_classes = y_true_bin.shape[1]

        # 计算每个类别的ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], self.y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 绘制每个类别的ROC曲线
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
        for i, color in zip(range(n_classes), colors):
            plt.figure(figsize=(12, 10))
            plt.plot(fpr[i], tpr[i], color=color, lw=3,
                     label=f'ROC曲线 (AUC = {roc_auc[i]:.5f})')  # 原始AUC标签
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

            # 放大的AUC标签（添加到曲线旁）
            plt.annotate(f'AUC = {roc_auc[i]:.5f}',
                         xy=(0.3, 0.6),  # 标签位置，可以根据需要调整
                         xytext=(0.3, 0.6),
                         fontsize=self.font_config.font_size + 10,  # 放大10号字体
                         fontweight=self.font_config.font_weight,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            # 设置坐标轴标签（增大字体）
            plt.xlabel('假阳性率 (FPR)', fontsize=self.font_config.font_size + 4,
                       fontweight=self.font_config.font_weight)
            plt.ylabel('真阳性率 (TPR)', fontsize=self.font_config.font_size + 4,
                       fontweight=self.font_config.font_weight)

            # 移除标题
            # plt.title(f'{self.labels_map[i]} 类别的 ROC 曲线', fontsize=self.font_config.font_size+4, fontweight=self.font_config.font_weight)

            # 保留图例但不显示AUC（因为已经在曲线上标注）
            plt.legend(loc="lower right",
                       prop={'weight': self.font_config.font_weight, 'size': self.font_config.font_size})

            plt.grid(True, linestyle='--', alpha=0.7)

            # 添加坐标网格线
            plt.minorticks_on()
            plt.grid(which='major', axis='both', linestyle='-', linewidth=0.5)
            plt.grid(which='minor', axis='both', linestyle=':', linewidth=0.3)

            # 保存图像
            plt.savefig(os.path.join(self.result_dir, f'ROC_{self.labels_map[i]}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        print(f"已生成 {len(self.labels_map)} 个类别ROC曲线")

    def plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        if self.y_true is None or self.y_score is None:
            print("请先执行评估")
            return

        # 计算混淆矩阵
        y_pred = np.argmax(self.y_score, axis=1)
        cm = confusion_matrix(self.y_true, y_pred)
        class_names = list(self.labels_map.values())

        # 可视化混淆矩阵
        plt.figure(figsize=(16, 14))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵', fontsize=self.font_config.font_size + 6, fontweight=self.font_config.font_weight)
        plt.colorbar(fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=self.font_config.font_size + 2,
                   fontweight=self.font_config.font_weight)
        plt.yticks(tick_marks, class_names, fontsize=self.font_config.font_size + 2,
                   fontweight=self.font_config.font_weight)

        # 在混淆矩阵上标注数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         verticalalignment="center",
                         fontsize=self.font_config.font_size + 2, fontweight=self.font_config.font_weight,
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('真实标签', fontsize=self.font_config.font_size + 4, fontweight=self.font_config.font_weight)
        plt.xlabel('预测标签', fontsize=self.font_config.font_size + 4, fontweight=self.font_config.font_weight)
        plt.savefig(os.path.join(self.result_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("已生成混淆矩阵")

    def print_classification_report(self):
        """打印分类报告"""
        if self.y_true is None or self.y_score is None:
            print("请先执行评估")
            return

        y_pred = np.argmax(self.y_score, axis=1)
        class_names = list(self.labels_map.values())

        print("\n分类报告:")
        print(classification_report(self.y_true, y_pred, target_names=class_names, digits=4))


# 主程序
if __name__ == "__main__":
    # 配置字体
    font_config = FontConfig(font_size=35, font_weight='bold')  # 可调整字体大小和粗细

    # 加载测试数据集
    try:
        test_dataset = ImageDataset(r'E:\WSI-HSfeature\CRC-VAL-HE-7K', transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print(f"测试数据集加载成功，共 {len(test_dataset)} 个样本")
    except Exception as e:
        print(f"加载测试数据集时出错: {e}")
        import sys

        sys.exit(1)

    # 加载模型
    model = models.resnet50(weights=None)
    # 修改最后一层以匹配我们的类别数
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 9)
    )

    try:
        model.load_state_dict(torch.load('E:\WSI-HSfeature/resnet50_weights.pth'))
        print("模型权重加载成功")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        sys.exit(1)

    model = model.to(device)
    model.eval()

    # 创建评估器并执行评估
    evaluator = ModelEvaluator(model, test_dataloader, result_dir, font_config)
    evaluator.evaluate()

    # 只生成ROC曲线
    evaluator.plot_roc_curves()

    # 注释掉不需要的函数调用
    # evaluator.plot_confusion_matrix()
    # evaluator.print_classification_report()

    print(f"ROC曲线已保存到 {result_dir} 文件夹中")

    #性能柱状图
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
    from sklearn.preprocessing import label_binarize
    from torchvision import transforms, models
    from torch.utils.data import DataLoader
    import os
    from PIL import Image
    from matplotlib.ticker import MaxNLocator

    # 设置字体为Arial并加粗
    plt.rcParams["font.family"] = ["Arial"]
    plt.rcParams["font.weight"] = "bold"

    # 创建结果保存目录
    result_dir = 'E:/WSI-HSfeature/9分类柱状图'
    os.makedirs(result_dir, exist_ok=True)

    # 使用指定颜色方案
    metric_color_map = {
        "Precision": "#2171A9",  # 指定蓝色
        "Recall": "#EF7B20",  # 指定橙色
        "F1-Score": "#329939",  # 指定绿色
    }

    # 类别标签映射
    class_labels = {0: "ADI", 1: "BACK", 2: "DEB", 3: "LYM", 4: "MUC",
                    5: "MUS", 6: "NORM", 7: "STR", 8: "TUM"}


    # 字体配置类
    class FontConfig:
        def __init__(self, label_size=28, tick_size=24, legend_size=24, annotation_size=22):
            self.label_size = label_size  # 坐标轴标签大小
            self.tick_size = tick_size  # 刻度标签大小
            self.legend_size = legend_size  # 图例大小
            self.annotation_size = annotation_size  # 注释大小


    # 定义数据集类
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, img_dir, transform=None, target_transform=None):
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

            imagelist = []
            lablelist = []
            # 检查img_dir是否存在
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Directory not found: {img_dir}")

            # 遍历所有子目录
            for folder in os.listdir(img_dir):
                folder_path = os.path.join(img_dir, folder)
                # 只处理目录
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file)
                        # 确保处理的是文件而不是子目录
                        if os.path.isfile(file_path):
                            imagelist.append(file)
                            lablelist.append(folder)

            self.imagelist = imagelist
            self.lablelist = lablelist
            print(f"Loaded {len(imagelist)} images, divided into {len(set(lablelist))} categories")

        def __len__(self):
            return len(self.imagelist)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.lablelist[idx], self.imagelist[idx])
            # 检查文件是否存在
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")

            image = Image.open(img_path)
            labels_map = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}
            label = labels_map[self.lablelist[idx]]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label


    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # 评估类
    class ModelEvaluator:
        def __init__(self, model, test_dataloader, result_dir, font_config=None):
            self.model = model
            self.test_dataloader = test_dataloader
            self.result_dir = result_dir
            self.font_config = font_config or FontConfig()
            self.y_true = None
            self.y_score = None
            self.y_pred = None

        def evaluate(self, save_predictions=True):
            """Perform model evaluation and save results"""
            # 获取模型预测
            self.y_true = []
            self.y_score = []

            with torch.no_grad():
                for inputs, labels in self.test_dataloader:
                    inputs = inputs.to(device)
                    outputs = self.model(inputs)
                    # 使用softmax获取概率
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    self.y_true.extend(labels.cpu().numpy())
                    self.y_score.extend(probs.cpu().numpy())

            self.y_true = np.array(self.y_true)
            self.y_score = np.array(self.y_score)
            self.y_pred = np.argmax(self.y_score, axis=1)  # 计算预测标签

            # 保存预测结果
            if save_predictions:
                np.save(os.path.join(self.result_dir, 'y_true.npy'), self.y_true)
                np.save(os.path.join(self.result_dir, 'y_score.npy'), self.y_score)
                np.save(os.path.join(self.result_dir, 'y_pred.npy'), self.y_pred)
                print(f"Prediction results saved to {self.result_dir}")

            return self.y_true, self.y_score, self.y_pred

        def calculate_performance_metrics(self):
            """Calculate performance metrics for each class"""
            if self.y_true is None or self.y_pred is None:
                print("Please run evaluation first")
                return None, None, None, None

            # 计算每个类别的精确率、召回率和F1分数
            precision = precision_score(self.y_true, self.y_pred, average=None)
            recall = recall_score(self.y_true, self.y_pred, average=None)
            f1 = f1_score(self.y_true, self.y_pred, average=None)
            accuracy = accuracy_score(self.y_true, self.y_pred)

            return precision, recall, f1, accuracy

        def plot_performance_bar_chart(self):
            """Plot performance metrics bar chart"""
            if self.y_true is None or self.y_pred is None:
                print("Please run evaluation first")
                return

            # 计算性能指标
            precision, recall, f1, accuracy = self.calculate_performance_metrics()
            if precision is None:
                return

            class_names = [class_labels[i] for i in range(9)]

            # 设置图形大小（宽高比24:12）
            plt.figure(figsize=(24, 12))
            plt.rcParams['figure.dpi'] = 300

            # 设置柱状图宽度
            bar_width = 0.22
            index = np.arange(len(class_names))

            # 绘制精确率、召回率和F1分数柱状图
            bar1 = plt.bar(index - bar_width, precision, bar_width, label='Precision',
                           color=metric_color_map["Precision"],
                           edgecolor='black', alpha=0.9, zorder=3)

            bar2 = plt.bar(index, recall, bar_width, label='Recall',
                           color=metric_color_map["Recall"],
                           edgecolor='black', alpha=0.9, zorder=3)

            bar3 = plt.bar(index + bar_width, f1, bar_width, label='F1-Score',
                           color=metric_color_map["F1-Score"],
                           edgecolor='black', alpha=0.9, zorder=3)

            # 添加坐标轴标签
            plt.xlabel('Class labels', fontsize=self.font_config.label_size, fontweight='bold')
            plt.ylabel('Scores', fontsize=self.font_config.label_size, fontweight='bold')
            plt.ylim(0, 1.05)  # 设置y轴范围

            # 设置x轴刻度和标签
            plt.xticks(index, class_names, rotation=0, ha='center', fontsize=self.font_config.tick_size)

            # 设置y轴刻度为小数形式
            plt.gca().yaxis.set_major_locator(MaxNLocator(prune='upper', nbins=6))

            # 添加网格线
            plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

            # 添加图例（放在图表上方）
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                       fancybox=True, shadow=True, ncol=3, framealpha=0.9,
                       prop={'weight': 'bold', 'size': self.font_config.legend_size})

            # 在柱状图上标注数值（保留两位小数）
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{height:.2f}', ha='center', va='bottom',
                             fontsize=self.font_config.annotation_size, fontweight='bold')

            add_labels(bar1)
            add_labels(bar2)
            add_labels(bar3)

            # 添加全局准确率
            plt.axhline(y=accuracy, color='r', linestyle='--', linewidth=2.5,
                        alpha=0.7, zorder=2)

            # 添加准确率标注
            plt.text(len(class_names) / 2, accuracy + 0.025,
                     f'Overall Accuracy: {accuracy:.2f}', color='r',
                     fontsize=self.font_config.legend_size, fontweight='bold',
                     ha='center', bbox=dict(facecolor='white', alpha=0.9, pad=6))

            # 优化边框显示
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            # 调整布局
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # 保存为PNG和PDF格式
            plt.savefig(os.path.join(self.result_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.result_dir, 'performance_metrics.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Performance metrics chart saved to {result_dir} (PNG and PDF)")

        def print_classification_report(self):
            """Print classification report"""
            if self.y_true is None or self.y_pred is None:
                print("Please run evaluation first")
                return

            class_names = [class_labels[i] for i in range(9)]

            print("\nClassification Report:")
            print(classification_report(self.y_true, self.y_pred, target_names=class_names, digits=4))


    # 主程序
    if __name__ == "__main__":
        # 定义设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')

        # 配置字体
        font_config = FontConfig()

        # 加载测试数据集
        try:
            test_dataset = ImageDataset(r'E:\WSI-HSfeature\CRC-VAL-HE-7K', transform=transform)
            test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            print(f"Test dataset loaded successfully, {len(test_dataset)} samples in total")
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            import sys

            sys.exit(1)

        # 加载模型
        model = models.resnet50(weights=None)
        # 修改最后一层以匹配我们的类别数
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 9)
        )

        try:
            model.load_state_dict(torch.load('E:\WSI-HSfeature/resnet50_weights.pth'))
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            sys.exit(1)

        model = model.to(device)
        model.eval()

        # 创建评估器并执行评估
        evaluator = ModelEvaluator(model, test_dataloader, result_dir, font_config)
        evaluator.evaluate()

        # 生成性能指标柱状图
        evaluator.plot_performance_bar_chart()

        # 打印分类报告
        evaluator.print_classification_report()

        print(f"Performance evaluation results saved to {result_dir}")