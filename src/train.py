from options import parse_args
from learning_algorithms import centralized, federated_optimizers, local_only
from utils.data_preprocessing import extract_images_from_videos, extract_amplitudes_from_images
from utils.general import send_message

import traceback
import logging
from PIL import ImageFile


def main():
    comm = False
    with open("token.txt") as file:
        token = file.read()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chat_id = "980853570"
    connection = (comm, url, chat_id)

    args = parse_args()
    print(args)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    try:
        if args.preprocess_data:
            extract_images_from_videos(width=args.width, height=args.height, data_path=args.data_root,
                                       heichole_video_path="I:/TrainingData/")
            if args.mode == "FedDG":
                extract_amplitudes_from_images(width=args.width, height=args.height, data_path=args.data_root)

        if args.mode == "centralized":
            centralized.main(args)
        elif args.mode == "local_only":
            local_only.main(args)
        else:
            federated_optimizers.main(args, connection)

    except:
        logging.error(traceback.format_exc())
        send_message(connection, traceback.format_exc())


if __name__ == "__main__":
    main()
