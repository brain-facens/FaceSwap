def create_model(opt):
    #from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
    from .fs_model import fsModel
    model = fsModel()
    model.initialize(opt)
    return model
