import torch
from tensorboardX import SummaryWriter
from datetime import datetime

class Trainer():

    def __init__(self, model, train_loader, val_loader, device):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        timestamp = datetime.timestamp(datetime.now())
        self.train_writer= SummaryWriter("runs/{}/train".format("mnist"))
        self.val_writer= SummaryWriter("runs/{}/val".format("mnist"))

        self.epoch = 0

    def train(self):

        self.model.train()

        criterion = torch.nn.CrossEntropyLoss()
        for i, sample in enumerate(self.train_loader):
            cur_iter = self.epoch*len(self.train_loader) + i

            if i % 50 == 0:
                self.validate(cur_iter)
                self.model.train()

            img, label = sample
            img = img.float().to(self.device)
            label = label.to(self.device)
            out = self.model(img)
            loss = criterion(out, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_writer.add_scalar('data/loss', loss.item(), cur_iter)
            print("Epoch: [{0}][{1}/{2}]\t Loss {3}".format(self.epoch, i, len(self.train_loader), round(loss.item(), 4)))

        self.epoch += 1

    def validate(self, cur_iter):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        for i, sample in enumerate(self.val_loader):
            img, label = sample
            img = img.float().to(self.device)
            label = label.to(self.device)

            out = self.model(img)
            loss = criterion(out, label)

            if i == 0:
                self.val_writer.add_scalar('data/loss', loss.item(), cur_iter)
                self.val_writer.add_image('img/image', img[0], cur_iter)
                print("Test: [{0}][{1}/{2}]\t Loss {3}".format(self.epoch, i, len(self.train_loader), round(loss.item(), 4)))



