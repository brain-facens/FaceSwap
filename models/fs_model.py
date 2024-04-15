import torch

from models.base_model import BaseModel
from models.fs_networks import Generator_Adain_Upsample

class fsModel(BaseModel):

    def initialize(self, opt) -> None:
        BaseModel.initialize(self, opt)
        torch.backends.cudnn.benchmark = True

        # device = torch.device("cuda:0")
        device = torch.device(device="cpu")

        # Generator network
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
        self.netG.to(device=device)

        # Id network
        netArc_checkpoint = opt.Arc_path
        netArc_checkpoint = torch.load(f=netArc_checkpoint, map_location=torch.device("cpu"))
        self.netArc = netArc_checkpoint
        self.netArc = self.netArc.to(device=device)
        self.netArc.eval()

        self.load_network(network=self.netG, network_label="G", epoch_label=opt.which_epoch, save_dir="")

    def forward(self, img_att, latent_id):
        img_fake = self.netG.forward(img_att, latent_id)
        return img_fake
