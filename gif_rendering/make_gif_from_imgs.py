import glob
from PIL import Image
import matplotlib.pyplot as plt
import pickle

def make_gif(frame_folder):
    frames =[Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    frame_one = frames[0]
    frame_one.save("GD.gif", format="GIF", append_images=frames,
               save_all=True, duration=500, loop=0)
make_gif("images/generated_steps")