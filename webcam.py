#######################################
# This code is made by BRAIN using
# SimSwap network to run in real-time
# with webcam.
#######################################

# IMPORTS
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.reverse2original import reverse2wholeimage
from torchvision.transforms import transforms
from options.test_options import TestOptions
from parsing_model.model import BiSeNet
from models.models import create_model
from util.norm import SpecificNorm
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
import numpy as np
import argparse
import qrcode
import torch
import cv2
import os


transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

opt = TestOptions().parse()


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=0, help='Define which usb port will be used to open the webcam.')
    arg = parser.parse_args()
    
    # variables
    is_running = True
    looking = True
    swap = False
    size = (480, 480)
    dev_port = 0
    port = arg.port
    limit = 5
    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size
    logoclass = ''
    file_path = 'FaceSwap_BRAIN.jpg'
    qr = qrcode.QRCode()

    # Set up the credentials
    # creds = Credentials.from_authorized_user_file('./google_credential/client_secret_685878340013-tq1op2etciprgf02lbr7882r6i66f4h7.apps.googleusercontent.com.json')
    # Set up the Drive API client
    # drive_service = build('drive', 'v3', credentials=creds)

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'

    model = create_model(opt)
    model.eval()
    mse = torch.nn.MSELoss().cuda()

    spNorm =SpecificNorm()

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=size, mode=mode)

    # checking available cameras
    if port == -1:
        while looking:
            camera = cv2.VideoCapture(dev_port)
            if camera.isOpened():
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    print(f'Port {dev_port} is \033[1;32mworking\033[0m and reads images ({h} x {w})')
                    looking = False
            if dev_port == limit:
                print('\033[1;31mNo camera available.\033[0m')
                exit(code=1)
            dev_port +=1
    else:
        camera = cv2.VideoCapture(port)
        if not camera.isOpened():
            print(f'\033[1;31mCamera {port} not available.\033[0m')
            exit(code=1)

    device =  torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current device: {device}')
    
    # reading webcam
    target_images = {}
    for image_file in Path('dataset').glob('*.jpg'):
        name = image_file.name.replace('_', ' ').replace('.jpg', '')
        target_face = cv2.imread(filename=str(image_file))
        target_face = cv2.resize(src=target_face, dsize=size)
        target_images[len(target_images)] = (name, target_face)

    base_target = target_face.copy()
    target_name = name
    index = len(target_images) - 1

    while is_running:

        # getting images
        if swap:
            image = swap_image.copy()
            cv2.putText(img=image, text='Press [c] to reset', org=(10,50), fontFace=1, fontScale=1, color=(255,255,0), thickness=2)
        else:
            # getting images
            check, image = camera.read()
            image = cv2.flip(src=image, flipCode=1)
            image = cv2.resize(src=image, dsize=size)
            base_image = image.copy()
            cv2.putText(img=image, text='Press [s] to swap', org=(10,50), fontFace=1, fontScale=1, color=(255,255,0), thickness=2)
        cv2.putText(img=target_face, text=f'Current Target: {target_images[index][0]}', org=(10,20), fontFace=1, fontScale=1, color=(0,255,0), thickness=2)
        cv2.putText(img=target_face, text='[a] - Previuos | [d] - Next', org=(10,50), fontFace=1, fontScale=1, color=(255,0,0), thickness=2)
        cv2.putText(img=image, text='Press [q] to exit', org=(10,20), fontFace=1, fontScale=1, color=(0,255,0), thickness=2)
        
        final_image = cv2.hconcat(src=[image, target_face]) # concating base and swap image
        final_image = cv2.resize(final_image, (1920,1080))

        # adding QR-Code
        #if swap:
        #    final_image = Image.fromarray(obj=final_image)
        #    final_image.paste(im=qr_image, box=(800,800))
        #    final_image = np.asarray(a=final_image)

        cv2.imshow(winname='FaceSwap', mat=final_image)

        key = cv2.waitKey(delay=1)

        if key == ord('q'):
            is_running = False
        
        elif key == ord('s') and swap == False:

            # base image
            img_a_align_crop, a_img_list = app.get(base_image, crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

            # convert numpy to tensor
            img_id = img_id.cuda()

            #create latent id
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)

            # target image
            specific_person_align_crop, b_specific_list = app.get(base_target, crop_size)
            specific_person_align_crop_pil = Image.fromarray(cv2.cvtColor(specific_person_align_crop[0],cv2.COLOR_BGR2RGB)) 
            specific_person = transformer_Arcface(specific_person_align_crop_pil)
            specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1], specific_person.shape[2])

            # convert numpy to tensor
            specific_person = specific_person.cuda()

            #create latent id
            specific_person_downsample = F.interpolate(specific_person, size=(112,112))
            specific_person_id_nonorm = model.netArc(specific_person_downsample)

            swap_result_list = []
            id_compare_values = [] 
            b_align_crop_tenor_list = []
            for b_align_crop in specific_person_align_crop:

                b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                b_align_crop_tenor_arcnorm = spNorm(b_align_crop_tenor)
                b_align_crop_tenor_arcnorm_downsample = F.interpolate(b_align_crop_tenor_arcnorm, size=(112,112))
                b_align_crop_id_nonorm = model.netArc(b_align_crop_tenor_arcnorm_downsample)

                id_compare_values.append(mse(b_align_crop_id_nonorm,specific_person_id_nonorm).detach().cpu().numpy())
                b_align_crop_tenor_list.append(b_align_crop_tenor)

            id_compare_values_array = np.array(id_compare_values)
            min_index = np.argmin(id_compare_values_array)
            min_value = id_compare_values_array[min_index]

            if opt.use_mask:
                n_classes = 19
                net = BiSeNet(n_classes=n_classes)
                net.cuda()
                save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
                net.load_state_dict(torch.load(save_pth))
                net.eval()
            else:
                net =None
            
            if min_value < opt.id_thres:

                swap_result = model(None, b_align_crop_tenor_list[min_index], latend_id, None, True)[0]

                swap_image = reverse2wholeimage([b_align_crop_tenor_list[min_index]], [swap_result], [b_specific_list[min_index]], crop_size, base_target, logoclass, \
                    os.path.join(opt.output_path, 'result_whole_swapspecific.jpg'), opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm, return_image=True)
                swap_image = cv2.resize(src=swap_image, dsize=size)
                # saving image
                cv2.imwrite(filename=file_path, img=swap_image)
                # # Upload the file to Drive
                # try:
                #     file_metadata = {'name': file_path}
                #     media = MediaFileUpload(file_path, resumable=True)
                #     file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                #     file_id = file.get('id')
                #     print(f'File ID: {file_id}')
                # except HttpError as error:
                #     print(f'An error occurred: {error}')
                # # Share the file with the public and get the public URL
                # try:
                #     permission = {'type': 'anyone', 'role': 'reader'}
                #     drive_service.permissions().create(fileId=file_id, body=permission).execute()
                #     file_url = f'https://drive.google.com/uc?id={file_id}'
                # except HttpError as error:
                #     print(f'An error occurred: {error}')
                # # generating QR-Code
                # qr.add_data(f'FaceSwap-BRAIN | {file_url}')
                #qr.add_data(f'FaceSwap-BRAIN | {target_name}\nURL')
                #qr.make()
                #qr_image = qr.make_image(fill_color='cyan', back_color='black').resize(size=(100,100))
                #qr_image = cv2.cvtColor(src=np.asarray(a=qr_image), code=cv2.COLOR_RGB2BGR)
                #qr_image = Image.fromarray(obj=qr_image)
                
                swap = True
            else:
                print(f'The person you specified is not found on the picture: {target_name}')

        elif key == ord('c') and swap == True:
            swap = False
        
        else:
            if (key == ord('a') or key == ord('d')) and not swap:
                if key == ord('a'):
                    index -= 1
                    if index < 0:
                        index = len(target_images) - 1
                elif key == ord('d'):
                    index += 1
                    if index == len(target_images):
                        index = 0
                target_name = target_images[index][0]
                target_face = target_images[index][1]
                base_target = target_face.copy()
        
    cv2.destroyAllWindows()
    exit(code=0)