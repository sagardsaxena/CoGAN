import sys
import logging
import torch
import torchvision
import torch.nn as nn
from dataset import *
import torchvision.transforms as transforms
from torch.autograd import Variable
from trainer_cogan import *
from net_config import *

from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from optparse import OptionParser
parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="net configuration",
                  default="../exps/mnist2usps_full_cogan.yaml")
parser.add_option('--gen_pkl', type=str, default=None)
parser.add_option('--dis_pkl', type=str, default=None)
parser.add_option('--iterations', type=int, default=0)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    assert isinstance(opts, object)
    config = NetConfig(opts.config)
    print(config)
    if os.path.exists(config.log):
        os.remove(config.log)
    base_folder_name = os.path.dirname(config.log)
    if not os.path.isdir(base_folder_name):
        os.mkdir(base_folder_name)
    logging.basicConfig(filename=config.log, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info("Let the journey begin!")
    logging.info(config)
    train_dataset_a = IMAGEDIR(root=config.train_data_a_path,
                               num_samples=config.train_data_a_size,
                               train=config.train_data_a_use_train_data,
                               label=0,
                               transform=transforms.ToTensor(),
                               seed=config.train_data_a_seed,
                               scale=config.scale,
                               bias=config.bias)
    train_loader_a = torch.utils.data.DataLoader(dataset=train_dataset_a, batch_size=config.batch_size, shuffle=True)

    train_dataset_b = IMAGEDIR(root=config.train_data_b_path,
                               num_samples=config.train_data_b_size,
                               train=config.train_data_b_use_train_data,
                               label=1,
                               transform=transforms.ToTensor(),
                               seed=config.train_data_b_seed,
                               scale=config.scale,
                               bias=config.bias)
    train_loader_b = torch.utils.data.DataLoader(dataset=train_dataset_b, batch_size=config.batch_size, shuffle=True)

    test_dataset_a = IMAGEDIR(root=config.test_data_a_path,
                               num_samples=config.test_data_a_size,
                               train=config.test_data_a_use_train_data,
                               label=0,
                               transform=transforms.ToTensor(),
                               seed=config.test_data_a_seed,
                               scale=config.scale,
                               bias=config.bias)
    test_loader_a = torch.utils.data.DataLoader(dataset=test_dataset_a, batch_size=config.test_batch_size, shuffle=True)

    test_dataset_b = IMAGEDIR(root=config.test_data_b_path,
                               num_samples=config.test_data_b_size,
                               train=config.test_data_b_use_train_data,
                               label=1,
                               transform=transforms.ToTensor(),
                               seed=config.test_data_b_seed,
                               scale=config.scale,
                               bias=config.bias)
    test_loader_b = torch.utils.data.DataLoader(dataset=test_dataset_b, batch_size=config.test_batch_size, shuffle=True)

    trainer = CoGanTrainer(config.batch_size, config.latent_dims)
    iterations = 0
    if opts.gen_pkl is not None and opts.dis_pkl is not None: 
        trainer.load_weights(opts.gen_pkl, opts.dis_pkl)
        print("Loaded", opts.gen_pkl, opts.dis_pkl, opts.iterations)
        iterations = opts.iterations

    trainer.cuda()
    dirname = os.path.dirname(config.snapshot_prefix)
    if not os.path.isdir(dirname):
      os.mkdir(dirname)
    dirname = os.path.join(dirname, "fakes")
    if not os.path.isdir(dirname):
      os.mkdir(dirname)

    from torch.nn.parameter import Parameter
    from torch.nn import MSELoss

    def lbfgs(img, lab):
      guess = [Parameter(torch.randn(len(img), config.latent_dims, requires_grad=True).cuda())]
      optimizer = torch.optim.LBFGS(guess)
      criterion = MSELoss()
      
      for i in range(10):
        def closure():
          optimizer.zero_grad()
          out = trainer.gen(guess[0])[lab]
          loss = criterion(out, img)
          loss.backward()
          return loss
        optimizer.step(closure)
      return guess[0]
      
    # Evaluate on TestA
    fake_b = os.path.join(dirname, "fake_b")
    if not os.path.isdir(dirname):
      os.mkdir(fake_b)

    for it, (imgs, labs) in enumerate(tqdm(test_loader_a)):
       res = lbfgs(imgs.cuda(), 0)
       
       fakes = trainer.gen(res)[1]
       for i, fake in enumerate(fakes):
           fname = "fakeb_{}_{}-{}.jpg".format(iterations, it, i)
           img_fname = os.path.join(fake_b, fname)
           torchvision.utils.save_image((fake.data-config.bias)/config.scale, img_fname) 
      
    # Evaluate on TestB
    fake_a = os.path.join(dirname, "fake_a")
    if not os.path.isdir(dirname):
      os.mkdir(fake_a)

    for it, (imgs, labs) in enumerate(tqdm(test_loader_b)):
       res = lbfgs(imgs.cuda(), 1)
       
       fakes = trainer.gen(res)[0]
       for i, fake in enumerate(fakes):
           fname = "fakea_{}_{}-{}.jpg".format(iterations, it, i)
           img_fname = os.path.join(fake_a, fname)
           torchvision.utils.save_image((fake.data-config.bias)/config.scale, img_fname) 
if __name__ == '__main__':
    main(sys.argv)

