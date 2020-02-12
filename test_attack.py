from argparse import ArgumentParser
import numpy as np
from attack import ModelSetAttack, PretrainedModelSetLoader, IndividualPrediction
import matplotlib.pyplot as plt
import tensorflow as tf
import configparser
from PIL import Image


def main(conf, args):
    print("Perform model set attack...")
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.7,
        visible_device_list="0")
    config = tf.ConfigProto(intra_op_parallelism_threads=0,
                            allow_soft_placement=True,
                            log_device_placement=False,
                            inter_op_parallelism_threads=0,
                            gpu_options=gpu_options,
                            use_per_session_threads=True)
    with tf.Session(config=config).as_default():
        with tf.device('/gpu:0'):
            img = Image.open('sample_1.png')
            x = np.ones((1, 224, 224, 3), dtype=np.float32)
            x[0] = np.array(img)
            pem = PretrainedModelSetLoader(conf, args)
            ip = IndividualPrediction(conf, args)
            msa = ModelSetAttack(conf, args)

            adv_x, stats = msa.generate(x)
            print(stats)
            model = pem.get_model()
            p_classes = ip(model.predict(x))
            print("p_classes: ", p_classes)
            pa_classes = ip(model.predict(adv_x))
            print("pa_classes: ", pa_classes)

            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(x.reshape(x.shape[1:]) / 255)
            axes[0].title.set_text('orig')
            pert = np.abs(adv_x - x).reshape(x.shape[1:]) / 255
            pert = (pert - np.min(pert)) / (np.max(pert) - np.min(pert))
            axes[1].imshow(pert)
            axes[1].title.set_text('pert')
            axes[2].imshow(adv_x.reshape(x.shape[1:]) / 255)
            axes[2].title.set_text('adv')
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Main entry point')

    parser.add_argument("-c", "--conf_file",
                        required=True,
                        help="Specify config file",
                        metavar="FILE")
    FLAGS, remaining_argv = parser.parse_known_args()
    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(FLAGS.conf_file)
    np.random.seed(9)
    main(conf, remaining_argv)
