import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import Vin2ParamLoss, Vin2ParamGRU


class Vin2ParamTrainer:
    def __init__(self, model: Vin2ParamGRU, train_dataloader: DataLoader,
                 val_dataloader: DataLoader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.opt = Adam(model.parameters())
        self.loss = Vin2ParamLoss()

    def train(self, epoch: int = 1, step_print: int = 50):
        for e in range(epoch):
            for i, (inp, label) in enumerate(self.train_dataloader):
                self.model.train()

                self.opt.zero_grad()

                out = self.model(inp)

                loss = self.loss(*out,
                                 label[:, 0],
                                 label[:, 1],
                                 label[:, 2])

                loss.backward()

                self.opt.step()

                if i % step_print == 0:
                    val_loss, brand_acc, model_acc, color_acc = self.eval(
                        self.model)
                    print(f"epoch: {e} | step: {i} | "
                          f"train_loss: {loss.item()} | "
                          f"val_loss: {val_loss} | "
                          f"brand_acc: {brand_acc} | "
                          f"model_acc: {model_acc} | "
                          f"color_acc: {color_acc}")

    def eval(self, model):
        model.eval()
        val_loss = 0
        brand_acc = 0
        model_acc = 0
        color_acc = 0
        for inp_val, label_val in self.val_dataloader:
            with torch.no_grad():
                out = model(inp_val)

                loss = self.loss(*out,
                                 label_val[:, 0],
                                 label_val[:, 1],
                                 label_val[:, 2])
            val_loss += loss.item()

            brand_out = torch.argmax(out[0], dim=-1)
            model_out = torch.argmax(out[1], dim=-1)
            color_out = torch.argmax(out[2], dim=-1)

            brand_acc += torch.sum(brand_out == label_val[:, 0]).item()
            model_acc += torch.sum(model_out == label_val[:, 1]).item()
            color_acc += torch.sum(color_out == label_val[:, 2]).item()

        val_size = len(self.val_dataloader) * 512
        val_loss /= len(self.val_dataloader)
        brand_acc /= val_size
        model_acc /= val_size
        color_acc /= val_size

        return val_loss, brand_acc, model_acc, color_acc
