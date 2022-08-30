import os
import base64
from glob import glob
import streamlit as st
import cv2, requests, json
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup
from PIL import Image
import time
import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from util import *
from PIL import Image
import math
import random
import os
import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
import lpips
import wandb
from model import *
from e4e_projection import projection as e4e_projection
from copy import deepcopy

os.makedirs('inversion_codes', exist_ok=True)
os.makedirs('style_images', exist_ok=True)
os.makedirs('style_images_aligned', exist_ok=True)
os.makedirs('models', exist_ok=True)

from enum import Enum

class FaceOption(Enum):
    arcane_caitlyn = 0
    arcane_jinx = 1
    disney = 2 
    jojo_yasuho = 3
    supergirl = 4
    goddess = 5
    def __str__(self):
        return f"{self.name}"
    @property
    def get_model(self):
        return os.path.join('models', f"{self.name}.pt")
    @property
    def get_img(self):
        return os.path.join('style_images_aligned', f"{self.name}.png")

    @classmethod
    def has_value(cls, value):
        return value in cls._member_names_   



download_with_pydrive = False 



if not os.path.isfile('models/dlibshape_predictor_68_face_landmarks.dat'):
    os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
    os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
    os.system('mv shape_predictor_68_face_landmarks.dat models/dlibshape_predictor_68_face_landmarks.dat')


drive_ids = {
    "stylegan2-ffhq-config-f.pt": "1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK",
    "e4e_ffhq_encode.pt": "1o6ijA3PkcewZvwJJ73dJ0fxhndn0nnh7",
    "restyle_psp_ffhq_encode.pt": "1nbxCIVw9H3YnQsoIPykNEFwWJnHVHlVd",
    "arcane_caitlyn.pt": "1gOsDTiTPcENiFOrhmkkxJcTURykW1dRc",
    "arcane_caitlyn_preserve_color.pt": "1cUTyjU-q98P75a8THCaO545RTwpVV-aH",
    "arcane_jinx_preserve_color.pt": "1jElwHxaYPod5Itdy18izJk49K1nl4ney",
    "arcane_jinx.pt": "1quQ8vPjYpUiXM4k1_KIwP4EccOefPpG_",
    "arcane_multi_preserve_color.pt": "1enJgrC08NpWpx2XGBmLt1laimjpGCyfl",
    "arcane_multi.pt": "15V9s09sgaw-zhKp116VHigf5FowAy43f",
    "disney.pt": "1zbE2upakFUAx8ximYnLofFwfT8MilqJA",
    "disney_preserve_color.pt": "1Bnh02DjfvN_Wm8c4JdOiNV4q9J7Z_tsi",
    "jojo.pt": "13cR2xjIBj8Ga5jMO7gtxzIJj2PDsBYK4",
    "jojo_preserve_color.pt": "1ZRwYLRytCEKi__eT2Zxv1IlV6BGVQ_K2",
    "jojo_yasuho.pt": "1grZT3Gz1DLzFoJchAmoj3LoM9ew9ROX_",
    "jojo_yasuho_preserve_color.pt": "1SKBu1h0iRNyeKBnya_3BBmLr4pkPeg_L",
    "supergirl.pt": "1L0y9IYgzLNzB-33xTpXpecsKU-t9DpVC",
    "supergirl_preserve_color.pt": "1VmKGuvThWHym7YuayXxjv0fSn32lfDpE",
    "art.pt": "1a0QDEHwXQ6hE_FcYEyNMuv5r5UnRQLKT",
    "goddess.pt": "1OcMunJBqfvHO4zgZ_7erQ5a54BJQ1lOT",
}

# from StyelGAN-NADA
class Downloader(object):
    def __init__(self, use_pydrive):
        self.use_pydrive = use_pydrive

        if self.use_pydrive:
            self.authenticate()
        
    def authenticate(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)
    
    def download_file(self, file_name):
        file_dst = os.path.join('models', file_name)
        file_id = drive_ids[file_name]
        if not os.path.exists(file_dst):
            print(f'Downloading {file_name}')
            if self.use_pydrive:
                downloaded = self.drive.CreateFile({'id':file_id})
                downloaded.FetchMetadata(fetch_all=True)
                downloaded.GetContentFile(file_dst)
            else:
                os.system(f'gdown --id {file_id} -O {file_dst}')
@st.cache
def get_model(device):
    # Load original generator
    latent_dim = 512
    original_generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    mean_latent = original_generator.mean_latent(10000)
    generator = deepcopy(original_generator)
    return generator
downloader = Downloader(False)

downloader.download_file('stylegan2-ffhq-config-f.pt')
downloader.download_file('e4e_ffhq_encode.pt')




from copy import deepcopy

# to be finetuned generator

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def norm_range(t, value_range):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))

def filter_img(img, my_w, option, device):
    ori_h, ori_w = img.shape[:2]
    ckpt = FaceOption[option].get_model
    try:
        downloader.download_file(os.path.basename(ckpt))
    except:
        pass
    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    generator = get_model(device)
    generator.load_state_dict(ckpt["g"], strict=False)
    generator.eval()

    with torch.no_grad():
        my_sample = generator(my_w, input_is_latent=True)
        norm_range(my_sample, (-1,1))
        img = np.array(my_sample.to('cpu')[0].permute(1,2,0).detach().cpu()*255, np.uint8)
        img = cv2.resize(img, (ori_w, ori_h))
            
    return img

    
if __name__ == '__main__':
    st.markdown("# ** Face to Face image translation Demo Page! **")
    st.markdown("### upload your own image and select style image to translate your style into. ")
    
    if torch.cuda.is_available():
        device = 'cuda'
#     else:
#         print('your torch is not available for cuda. this will make the entire process slower')
#     device='cpu'
    
    gan_option = st.radio('Select option:',list(FaceOption.__members__))
    inputspace = st.columns([0.75, 0.25])
    ori_img = inputspace[0].file_uploader(label='source image, jpg | png | jpeg | webp', type=['png', 'jpg', 'jpeg', 'webp'])
    rotate = st.radio('rotate picture. clock-wise :',['0', '90', '180', '270'])
    with st.form("input_form"):
        if ori_img is not None:
            img_col = st.columns([1,1])
            img_col[0].write("**uploaded image**")
            img_col[1].write("**reference image**")
            ori_img = np.array(Image.open(ori_img))
            if rotate == '90':
                img = cv2.rotate(ori_img, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == '180':
                img = cv2.rotate(ori_img, cv2.ROTATE_90_CLOCKWISE)
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == '270':
                img = cv2.rotate(ori_img, cv2.ROTATE_90_CLOCKWISE)
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                img = ori_img
            img = Image.fromarray(img)
            img_col[0].image(img, channels = "BGR")
            img_col[1].image(Image.open(FaceOption[gan_option].get_img), channels = "BGR")
            original_img = img.convert("RGB")
            
            # this will save uploaded image, you will have to
            cnt = 0

            file_path = None
            while True:
                file_path = f"save/temp{str(cnt).zfill(4)}.png"

                if os.path.isfile(file_path):
                    cnt+=1
                    continue
                break
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            original_img.save(file_path)
            with open('save_path.txt', 'w') as f:
                f.write(file_path)
           
        style_cols = st.columns([1, 1.5])
        st.sidebar.write("# ***Style Photo Preview***")
        for item in list(FaceOption):
            st.sidebar.write(f"#### **{item}**")
            st.sidebar.image(item.get_img)
        submitted = st.form_submit_button("Process")
    if submitted:
        with open('save_path.txt', 'r') as f:
            file_path = f.read()
        aligned_face = align_face(file_path)

        assert aligned_face is not None, "Face not detected"
        
        my_w = e4e_projection(aligned_face, '원본', device).unsqueeze(0)
        bg_image = filter_img(cv2.cvtColor(np.array(aligned_face), cv2.COLOR_BGR2RGB), my_w,gan_option, device)
        img_col = st.columns([1,1])
        img_col[0].write("**Original**")
        img_col[1].write("**Filtered**")
        img_col[0].image(np.array(aligned_face), channels = "RGB")
        img_col[1].image(bg_image, channels = "RGB")
                        

