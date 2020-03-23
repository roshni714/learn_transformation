import torch
from tensorboardX import SummaryWriter
from datetime import datetime

class TransformNetTrainer():

    def __init__(self, transform_net, transform_list, pretrained_model, train_loader, test_loader, device):
        print("Setting up TransformNetTrainer")

        self.device = device
        self.transform_net = transform_net
        self.pretrained_model = pretrained_model
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = torch.optim.Adam(self.transform_net.parameters(), lr=1e-2)

        names ='_'.join(transform_list)

        self.transform_list = transform_list
        timestamp = datetime.timestamp(datetime.now())
        self.writer= SummaryWriter("runs/{}/{}".format(names, timestamp))

        self.epoch = 0
        self.train_data_embeddings = self.generate_train_embeddings()


    def generate_train_embeddings(self):
        print("Generating trainset embeddings")
        embeddings = []

        for i, sample in enumerate(self.train_loader):
            img, label = sample
            img = img.float().to(self.device)
            embeddings.append(self.pretrained_model(img).detach())

        return embeddings


    def nearest_neighbor_embedding_loss(self, test_example_batch):

        self.pretrained_model.eval()
        test_embedding_batch = self.pretrained_model(test_example_batch)
        
        mse= torch.nn.MSELoss(reduction="sum")

        batch_loss = 0.
        for j in range(test_example_batch.shape[0]):
            min_dist = float("inf")
            for train_embedding in self.train_data_embeddings:
                loss = mse(train_embedding, test_embedding_batch[j])

                if loss < min_dist:
                    min_dist = loss

            batch_loss += min_dist

        return batch_loss/test_example_batch.shape[0]

    def train(self):

        self.pretrained_model.eval()
        self.transform_net.train()


        for i, sample in enumerate(self.test_loader):
            cur_iter = self.epoch*len(self.test_loader) + i

            img, label = sample
            img = img.float().to(self.device)
            
            transform_out  = self.transform_net(img)
            transformed_test_batch, transform_param= self.apply_transform_batch(img, transform_out)
            loss = self.nearest_neighbor_embedding_loss(transformed_test_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('data/loss', loss.item(), cur_iter)
            for j, tf  in enumerate(self.transform_list):
                self.writer.add_scalar('data/{}'.format(tf), transform_param[j].item(), cur_iter)
                self.writer.add_image('img/corrupted', img[0], cur_iter)
                self.writer.add_image('img/transformed', transformed_test_batch[0], cur_iter)
            print("Epoch: [{0}][{1}/{2}]\t Loss {3}".format(self.epoch, i, len(self.test_loader), round(loss.item(), 4)))
        self.epoch += 1


    def apply_transform_batch(self, img_batch, transform_out):
        transform_act = torch.zeros(1, transform_out.shape[-1])
        sigmoid = torch.nn.Sigmoid()
        tanh = torch.nn.Tanh()

        new_img_batch = torch.zeros(img_batch.shape).to(img_batch.get_device())
        for i in range(img_batch.shape[0]):
            for j, tf in enumerate(self.transform_list):
                name = tf

                if name == "saturation":
                    sat_act = 2 * sigmoid(transform_out[i][j])
                    transform_act[j] += sat_act
                    new_img_batch[i] = adjust_saturation(img_batch[i], sat_act)
                if name == "brightness":
                    bright_act = tanh(transform_out[i][j])
                    transform_act[j] += bright_act
                    new_img_batch[i] = adjust_brightness(img_batch[i], bright_act)

        transform_act /= new_img_batch.shape[0]
        return new_img_batch, transform_act
            

def adjust_saturation(img, sat):

    device = img.get_device()
    square = torch.pow(img, 2)
    vals = [0.299, 0.587, 0.114]
    mult = torch.ones(img.shape)
    for i in range(len(vals)):
        mult[i, :, :] *= vals[i]
    mult= mult.to(device)
    res = square * mult
    p = res.sum(dim=0).sqrt().unsqueeze(0)

    copy_p = p.repeat(3, 1, 1)

    new_img = copy_p + (img - copy_p) * sat
    return new_img

def adjust_brightness(img, brightness):
    new_img = img + brightness
    return new_img


