import gc
import os
import random
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageFile

from miscellaneous.scd_delivery.main.scd_modules.signature_net import SignatureNet1
from miscellaneous.scd_delivery.main.scd_modules.similarity_net import SimilarityNet

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class ServicesSCD:
    def __init__(self):
        """
        Service methods for creation of camera signatures, and compare them using correlation
        """
        pre_trained_models = Path(os.path.dirname(os.path.realpath(__file__))).joinpath('pre_trained_models')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Attributes concerning the signature network
        self.sig_net_trained_model = pre_trained_models.joinpath('signature_net.pt')
        self.sig_net = None
        self.sig_size = 1024

        # Attributes concerning the similarity network
        self.sim_net_trained_model = pre_trained_models.joinpath('similarity_net.pt')
        self.sim_net = None

    @staticmethod
    def validate_and_load_image(image_path):
        """
        Auxiliary function to load and check whether an image, given in terms of
        an image path, is supported or not.
        """
        try:
            with Image.open(image_path) as image:
                if image is None:
                    raise ValueError("Invalid image file")

                # validation of the image format
                original_image_ext = str.lower(image.format)
                file_ext = os.path.basename(image_path).split('.')[-1].lower()
                valid_ext = ['jpeg', 'jpg', 'bmp', 'png', 'gif']

                if file_ext not in valid_ext:
                    error_message = "Invalid image extension: allowed extensions are: .png, .jpg, .jpeg, .bmp, .gif"
                    raise ValueError(error_message)

                if file_ext in {'jpg', 'jpeg'}:
                    if original_image_ext not in {'jpg', 'jpeg'}:
                        raise ValueError('Mismatch between actual image type and image name')
                elif file_ext != original_image_ext:
                    raise ValueError('Mismatch between actual image type and image name')

                if image.width < 128 or image.height < 128:
                    raise ValueError('Image dimensions (width x height x channels) must be minimum 128x128x3')

                if image.mode == 'P':
                    image = image.convert('RGBA').convert('RGB')
                elif image.mode != 'RGB':
                    image = image.convert('RGB')

                return image.copy()

        except Exception as e:
            raise ValueError(str(e))

    def load_signature_net1(self):
        """
        Loads the signature_net model into the memory. This method needs to be called before the call to
        compute_correlation_similarity_net1. It is recommended to run this method as few times as possible.
        There are no inputs and no outputs for this method.
        WARNING: Repeated usage may lead to memory exceptions
        """
        self.sig_net = SignatureNet1(num_classes=58)
        self.sig_net.load_state_dict(torch.load(self.sig_net_trained_model))
        self.sig_net = self.sig_net.to(self.device)
        self.sig_net.eval()

    @staticmethod
    def extract_patches(img_data, std_threshold=0.02, max_num_patches=1):
        """
        Extracts patches to extract features signature using net1
        """
        patches = []

        patch = namedtuple('WindowSize', ['width', 'height'])(128, 128)
        stride = namedtuple('Strides', ['width_step', 'height_step'])(128, 128)
        image = namedtuple('ImageSize', ['width', 'height'])(img_data.shape[1], img_data.shape[0])
        num_channels = 3

        # Choose the patches
        for row_idx in range(patch.height, image.height, stride.height_step):
            for col_idx in range(patch.width, image.width, stride.width_step):
                cropped_img = img_data[(row_idx - patch.height):row_idx, (col_idx - patch.width):col_idx]
                patch_std = np.std(cropped_img.reshape(-1, num_channels), axis=0)
                if np.prod(np.less_equal(patch_std, std_threshold)):
                    patches.append(cropped_img)

        # Filter out excess patches
        if len(patches) > max_num_patches:
            random.seed(999)
            indices = random.sample(range(len(patches)), max_num_patches)
            patches = [patches[x] for x in indices]

        if len(patches) == 0:
            patches = [img_data[0:patch.height, 0:patch.height]]

        return patches

    def extract_features_signature_net1(self, input_image_full_filename: str) -> np.ndarray:
        """
        The method computes the camera signature from the input image. Scene content from the image is suppressed using
        a pre-trained Convolutional Neural Network (CNN). The residual image on suppressing the scene content captures
        the sensor noise characteristics. The features of this residual image are extracted and will be called the
        camera signature. CNN is designed and trained based on the architectures proposed in [1] and [2].

        :param input_image_full_filename: RGB image file name (with full path)
        :type input_image_full_filename: str. Support files are:
            * JPEG files - *.jpeg, *.jpg.
            * Portable Network Graphics - *.png

        :return: output_camera_signature
        :rtype: 2D numpy array
            * type: float64
            * size: varying between <1,1024> and <20, 1024>

        .. note::
            The output is a 2D numpy array whose size varies depending on the image content. For images with several
            homogeneous regions the feature vector would be of size <20, 1024>, for images with no homogeneous regions
            the feature vector would be of size <1, 1024>. For all other images, the feature vector size can vary
            between <1, 1024> to <20, 1024>.

        .. [1]  Zhang, Kai, et al. "Beyond a gaussian denoiser: Residual learning of deep CNN for image denoising". In
                IEEE Transactions on Image Processing 26.7 (2017): 3142-3155.

        .. [2]	Mayer, Owen, and Matthew C. Stamm. "Forensic Similarity for Digital Images". In arXiv preprint
                arXiv:1902.04684 (2019).
        """

        # Check for the input type : must be string
        if type(input_image_full_filename) is not str:
            raise ValueError("Expected 'str' input, but found {}".format(str(type(input_image_full_filename))))

        # Check for valid path
        if not os.path.isfile(input_image_full_filename):
            raise ValueError("Non-existent file path : {}".format(input_image_full_filename))

        img = self.validate_and_load_image(input_image_full_filename)
        img = np.asarray(img) / 255.0

        # extract patches and subtract patch mean
        patches = self.extract_patches(img)
        img = np.stack(patches, axis=0)
        img = img - np.mean(img, axis=(1, 2), keepdims=True)
        img = np.transpose(img, (0, 3, 2, 1))  # (b,w,h,c) --> (b,c,h,w)
        img = torch.tensor(img, dtype=torch.float32, device=self.device)

        # Predict
        signature = self.sig_net.extract_features(img)
        signature = signature.to(torch.device("cpu")).detach().numpy()
        return signature

    def unload_signature_net1(self):
        """
        Attempt to unload the signature_net model from memory.
        This method should be called at the end of the user session.
        Note that, every call to the method load_signature_net1 must end with a matching call to unload_signature_net1.
        """
        self.sig_net = None
        gc.collect()

    def load_similarity_net1(self):
        """
        Loads the similarity_net model into the memory.
        This method needs to be called before the call to
        compute_correlation_similarity_net1. It is recommended to run this method as few times as possible.
        There are no inputs and no outputs for this method.
        WARNING: Repeated usage may lead to memory exceptions
        """
        self.sim_net = SimilarityNet()
        self.sim_net.load_state_dict(torch.load(self.sim_net_trained_model))
        self.sim_net = self.sim_net.to(self.device)
        self.sim_net.eval()

    def compute_correlation_similarity_net1(
            self, input_query_img_signature: np.ndarray, input_ref_img_signature: np.ndarray,
            input_threshold: float = 0.50) -> Tuple[float, bool]:
        """
        This function receives one query signature and another reference image signature (pre-computed for different
        images). A similarity score is generated. The greater the score, the higher the probability indicating the same
        source for the provided query and reference images. The architecture of the similarity network is designed based
        on the work  in [2] and [3]. An optional parameter “input_threshold” can be used to decide the threshold above
        which the query and the reference signatures are considered to be similar. The default value of this parameter
        is set to 0.50 (We use the threshold value of 0.95 for investigations on the Dresden dataset
        http://forensics.inf.tu-dresden.de/ddimgdb/).

        :param input_query_img_signature: signature of the query image computed using the compute_signature_ResidualNet1
        :type input_query_img_signature:
            * type: 2D numpy array with 64 bit float values
            * length/size: varying between <1,1024> and <20, 1024>

        :param input_ref_img_signature: signature of the reference img computed using the compute_signature_ResidualNet1
        :type input_ref_img_signature:
            * type: 2D numpy array with 64 bit float values
            * length/size: varying between <1,1024> and <20, 1024>

        :param input_threshold: An optional parameter “input_threshold” can be used to decide the threshold above which
        the query and the reference signatures are considered to be similar. The default value of this parameter is 0.50
        :type input_threshold:
            * type: float64
            * length: 1
            * range: [0,1]

        :return: output_correlation_score
        :rtype:
            * type: bool
            * The output will be True if the two input signatures are similar, that is when the output_score is above
            the input_threshold. The output will be False otherwise.

        .. note::
            The inputs are a 2D numpy array whose size varies depending on the image content. For images with several
            homogeneous regions the feature vector would be of size <20, 1024>, for images with no homogeneous regions
            the feature vector would be of size <1, 1024>. For all other images, the feature vector size can vary
            between <1, 1024> to <20, 1024>.

        .. [2]	Mayer, Owen, and Matthew C. Stamm. "Forensic Similarity for Digital Images". In arXiv preprint
                arXiv:1902.04684 (2019).

        .. [3]	Mayer, Owen, and Matthew C. Stamm. "Learned forensic source similarity for unknown camera models". In
                2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.
        """

        # Checking on the input types
        if type(input_query_img_signature) is not np.ndarray:
            raise ValueError('Query signature must be of type numpy.ndarray')
        if type(input_ref_img_signature) is not np.ndarray:
            raise ValueError('Reference signature must be of type numpy.ndarray')

        # Checking on the dimensions of the inputs
        assert input_query_img_signature.shape[-1] == self.sig_size, f'Query signature size must be {self.sig_size}'
        assert input_ref_img_signature.shape[-1] == self.sig_size, f'Reference signature size must be {self.sig_size}'

        correlation_scores = []
        for ref_sig in input_ref_img_signature:
            for query_sig in input_query_img_signature:
                signature_pair = [np.reshape(ref_sig, (1, -1)), np.reshape(query_sig, (1, -1))]
                signature_pair = [torch.tensor(x, device=self.device) for x in signature_pair]
                correlation_scores.append(self.sim_net(signature_pair).detach().item())

        score = float(np.mean(correlation_scores))
        if score >= input_threshold:
            correlation_prediction = True
        else:
            correlation_prediction = False

        return score, correlation_prediction

    def unload_similarity_net1(self):
        """
        Attempt to unload the similarity_net model from memory.
        This method should be called at the end of the user session.
        Note that, every call to the method load_similarity_net1 must end with a matching call to unload_similarity_net1
        """
        self.sim_net = None
        gc.collect()
