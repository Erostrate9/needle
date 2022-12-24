import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
from models import ResNet9
from simple_training import train_cifar10, evaluate_cifar10

device = ndl.cuda()
dataset = ndl.data.CIFAR10Dataset("../data/cifar-10-batches-py", train=True)
dataloader = ndl.data.DataLoader(
         dataset=dataset,
         batch_size=128,
         shuffle=True,
         # collate_fn=ndl.data.collate_ndarray,
         device=device,
         dtype="float32")
model = ResNet9(device=device, dtype="float32")
train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
      lr=0.001, weight_decay=0.001)
evaluate_cifar10(model, dataloader)

# train epoch0: avg_acc: 0.3889, avg_loss: [0.01331361], time cost: 42.357852935791016
# train epoch1: avg_acc: 0.49306, avg_loss: [0.01093788], time cost: 42.02511429786682
# train epoch2: avg_acc: 0.54246, avg_loss: [0.00996746], time cost: 41.36898422241211
# train epoch3: avg_acc: 0.57676, avg_loss: [0.00925622], time cost: 41.449687242507935
# train epoch4: avg_acc: 0.60182, avg_loss: [0.00868693], time cost: 41.34904098510742
# train epoch5: avg_acc: 0.6285, avg_loss: [0.00816438], time cost: 41.6195867061615
# train epoch6: avg_acc: 0.647, avg_loss: [0.00772511], time cost: 42.19674515724182
# train epoch7: avg_acc: 0.66982, avg_loss: [0.00732698], time cost: 41.557111501693726
# train epoch8: avg_acc: 0.68548, avg_loss: [0.00697118], time cost: 41.48004627227783
# train epoch9: avg_acc: 0.6988, avg_loss: [0.0066537], time cost: 44.75082969665527
# evaluate: avg_acc: 0.66462, avg_loss: [0.00742565]