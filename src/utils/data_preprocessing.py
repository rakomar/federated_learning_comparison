import numpy as np
from tqdm import tqdm
from PIL import Image
import glob
import shutil
import pims
from utils.save_results import clear_directory


def extract_images_from_videos(width, height, data_path, heichole_video_path):
    for video_num in range(1, 25):
        print(f"Extracting images from HeiChole Surgery {str(video_num)}")
        filename = f"{heichole_video_path}/HeiChole{str(video_num)}.mp4"
        video = pims.Video(filename)
        print(video.frame_rate)

        # create storage directory
        clear_directory(f"{data_path}/train/HeiChole/{str(video_num)}")
        srcpath = f"{heichole_video_path}/Annotations/Instrument/Hei-Chole{str(video_num)}_Annotation_Instrument.csv"
        dstpath = f"{data_path}/train/HeiChole/{str(video_num)}/Ins.csv"
        shutil.copyfile(srcpath, dstpath)

        removed_frames = []

        for frame_num in tqdm(range(0, len(video), int(video.frame_rate)), desc="Frames"):
            frame = Image.fromarray(video[frame_num])

            if frame_num < 2000:
                continue

            import matplotlib.pyplot as plt
            save_dir = "G:/Studium/MasterCMS/MastersThesis/Report Bilder"
            print(frame_num)
            plt.imshow(frame)
            plt.axis("off")
            fig = plt.gcf()
            fig.savefig(f"{save_dir}/frame.pdf",
                        bbox_inches='tight',
                        dpi=600)
            plt.show()

            frame = frame.resize(size=(width, height), resample=Image.NEAREST)

            # remove white frames
            if np.min(frame) == 255:
                removed_frames.append(frame_num)
                continue

            # save frame
            frame.save(f"{data_path}/train/HeiChole/" + str(video_num) + "/" + str(frame_num).zfill(8) + ".png")

        # create file to track indices of white (removed) frames, used to skip corresponding annotations
        with open(f"{data_path}/train/HeiChole/" + str(video_num) + "/removed_frames.txt", "w") as file:
            file.write(str(removed_frames))


def extract_amplitude_spectrum(image):
    """
    Compute the amplitude spectrum of an image
    """
    fft = np.fft.fft2(image, axes=(-2, -1))
    amplitude = np.abs(fft)
    return amplitude


def extract_amplitudes_from_images(width, height, data_path):
    # HeiChole
    for video_num in range(1, 25):
        print("Extracting amplitudes from HeiChole Surgery " + str(video_num))

        clear_directory(f"{data_path}/train/HeiChole/{str(video_num)}/amplitudes")
        frame_names = glob.glob(f"{data_path}/train/HeiChole/{str(video_num)}/*.png")

        for name in frame_names:
            frame = Image.open(name)
            frame = np.asarray(frame)
            frame = frame.transpose((2, 0, 1))  # transpose to make compatible with AmplitudeInterpolation
            amp = extract_amplitude_spectrum(frame).astype(np.float32)
            filename = name.split("\\")[1][:-4]
            np.save(f"{data_path}/train/HeiChole/{str(video_num)}/amplitudes/amp_{filename}", amp)

    # Cholec80
    for video_num in list(range(1, 81)):
        print("Extracting amplitudes from Cholec80 Surgery " + str(video_num))

        clear_directory(f"{data_path}/train/Cholec80/{str(video_num)}/amplitudes")
        frame_names = glob.glob(f"{data_path}/train/Cholec80/{str(video_num)}/*.png")

        for name in frame_names:
            frame = Image.open(name)
            frame = frame.resize(size=(width, height), resample=Image.NEAREST)
            frame = np.asarray(frame)
            frame = frame.transpose((2, 0, 1))  # transpose to make compatible with AmplitudeInterpolation

            amp = extract_amplitude_spectrum(frame).astype(np.float32)
            filename = name.split("\\")[1][:-4]
            np.save(f"{data_path}/train/Cholec80/{str(video_num)}/amplitudes/amp_{filename}.npy", amp)
