import os
import unittest

import numpy as np

from miscellaneous.scd_delivery.main.scd_modules.services import ServicesSCD


class TestComputeSignature(unittest.TestCase):

    def setUp(self) -> None:
        self.scd_services = ServicesSCD()
        self.scd_services.load_signature_net1()
        self.test_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unit_test_input_files')
        self.num_decimal_digits_accuracy = 2  # The number of decimal places to account for accuracy

    def tearDown(self) -> None:
        self.scd_services.unload_signature_net1()

    def test_SCD_extract_features_signature_net1_001(self) -> None:
        """
        SCD_extract_features_signature_net1_001: Extract signature from .jpg image
        """
        expected_output = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid_jpg.npy'))
        jpg_img_path = os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid.JPG')
        signature = self.scd_services.extract_features_signature_net1(jpg_img_path)
        np.save(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid_jpg.npy'), signature)
        max_difference = np.max(np.abs(expected_output - signature))
        self.assertAlmostEqual(max_difference, 0.0, places=self.num_decimal_digits_accuracy)

    def test_SCD_extract_features_signature_net1_002(self) -> None:
        """
        SCD_extract_features_signature_net1_002: Extract signature from .jpeg image
        """
        expected_output = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid_jpeg.npy'))
        jpeg_img_path = os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid.jpeg')
        signature = self.scd_services.extract_features_signature_net1(jpeg_img_path)
        np.save(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid_jpeg.npy'), signature)
        max_difference = np.max(np.abs(expected_output - signature))
        self.assertAlmostEqual(max_difference, 0.0, places=self.num_decimal_digits_accuracy)

    def test_SCD_extract_features_signature_net1_003(self) -> None:
        """
        SCD_extract_features_signature_net1_003: Extract signature from .png image
        """
        expected_output = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid_png.npy'))
        png_img_path = os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid.png')
        signature = self.scd_services.extract_features_signature_net1(png_img_path)
        np.save(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid_png.npy'), signature)
        max_difference = np.max(np.abs(expected_output - signature))
        self.assertAlmostEqual(max_difference, 0.0, places=self.num_decimal_digits_accuracy)

    def test_SCD_extract_features_signature_net1_004(self) -> None:
        """
        SCD_extract_features_signature_net1_004: Extract signature from empty path
        """
        empty_path = ''
        with self.assertRaises(ValueError) as err:
            self.scd_services.extract_features_signature_net1(empty_path)
        self.assertTrue(str(err.exception).find("Non-existent file path : ") != -1)

    def test_SCD_extract_features_signature_net1_005(self) -> None:
        """
        SCD_extract_features_signature_net1_005: Extract signature from renamed .jpg file (like renamed .gif or
        .bmp as .jpg)
        """
        invalid_jpg_file = os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_invalid.jpg')
        with self.assertRaises(ValueError) as err:
            self.scd_services.extract_features_signature_net1(invalid_jpg_file)
        self.assertTrue((str(err.exception) == "Mismatch between actual image type and image name") != -1)

    def test_SCD_extract_features_signature_net1_006(self) -> None:
        """
        SCD_extract_features_signature_net1_006: Extract signature from renamed .jpeg file (like renamed .gif or
        .bmp as .jpeg)
        """
        invalid_jpeg_file = os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_invalid.jpeg')
        with self.assertRaises(ValueError) as err:
            self.scd_services.extract_features_signature_net1(invalid_jpeg_file)
        self.assertTrue((str(err.exception) == "Mismatch between actual image type and image name") != -1)

    def test_SCD_extract_features_signature_net1_007(self) -> None:
        """
        SCD_extract_features_signature_net1_007: Extract signature from renamed .png file (like renamed .gif or
        .bmp as .png)
        """
        invalid_png_file = os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_invalid.png')
        with self.assertRaises(ValueError) as err:
            self.scd_services.extract_features_signature_net1(invalid_png_file)
        self.assertTrue((str(err.exception) == "Mismatch between actual image type and image name") != -1)

    def test_SCD_extract_features_signature_net1_008(self) -> None:
        """
        SCD_extract_features_signature_net1_008: Extract signature from any non-valid file (all files except jpeg,
        png and jpeg files)
        """
        text_file_path = os.path.join(self.test_files_dir, 'test_text_file.txt')
        with self.assertRaises(ValueError) as err:
            self.scd_services.extract_features_signature_net1(text_file_path)
        self.assertTrue(str(err.exception).find("cannot identify image file") != -1)

    def test_SCD_extract_features_signature_net1_009(self) -> None:
        """
        SCD_extract_features_signature_net1_009: Extract signature from invalid path
        """
        invalid_path = os.path.join(self.test_files_dir)
        with self.assertRaises(ValueError) as err:
            self.scd_services.extract_features_signature_net1(invalid_path)
        self.assertTrue(str(err.exception).find("Non-existent file path : ") != -1)

    def test_SCD_extract_features_signature_net1_010(self) -> None:
        """
        SCD_extract_features_signature_net1_010: Extract signature from non-string type inputs
        """
        with self.assertRaises(ValueError) as err:
            self.scd_services.extract_features_signature_net1([1, 2, 3])
        self.assertTrue(str(err.exception).find("Expected 'str' input, but found ") != -1)

    def test_SCD_extract_features_signature_net1_011(self) -> None:
        """
        SCD_extract_features_signature_net1_011: Load and unload the model
        """
        try:
            self.scd_services.load_signature_net1()
            self.scd_services.unload_signature_net1()
            self.setUp()
        except Exception as err:
            self.fail("Error in loading and unloading of signature network")

    # continuous predictions
    def test_SCD_extract_features_signature_net1_012(self) -> None:
        """
        SCD_extract_features_signature_net1_012: Test the method on 5 different inputs by consecutive evaluation
        """
        # Initialize the placeholders
        num_tests = 5
        expected_output = [np.empty(0)] * num_tests
        img_path = [""] * num_tests
        signature = [np.empty(0)] * num_tests

        # Prepare the inputs
        expected_output[3] = expected_output[0] = np.load(
            os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid_jpg.npy'))
        img_path[3] = img_path[0] = os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid.JPG')

        expected_output[4] = expected_output[1] = np.load(
            os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid_jpeg.npy'))
        img_path[4] = img_path[1] = os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid.jpeg')

        expected_output[2] = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid_png.npy'))
        img_path[2] = os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid.png')

        # Run the tests
        for i in range(num_tests):
            signature[i] = self.scd_services.extract_features_signature_net1(img_path[i])

        # Evaluate the results
        for i in range(num_tests):
            max_difference = np.max(np.abs(expected_output[i] - signature[i]))
            self.assertAlmostEqual(max_difference, 0.0, places=self.num_decimal_digits_accuracy)

    def test_SCD_extract_features_signature_net1_013(self) -> None:
        """
        SCD_extract_features_signature_net1_013: Test minimum image size 128x128x3
        """
        jpeg_img_path = os.path.join(self.test_files_dir, 'Invalid_image_size.jpg')
        with self.assertRaises(ValueError) as err:
            self.scd_services.extract_features_signature_net1(jpeg_img_path)
        self.assertTrue(
            str(err.exception).find("Image dimensions (width x height x channels) must be minimum 128x128x3") != -1)

    def test_SCD_extract_features_signature_net1_014(self) -> None:
        """
        SCD_extract_features_signature_net1_014: Extract signature from .gif image
        """
        expected_output = np.load(os.path.join(self.test_files_dir, 'gif_image_valid.npy'))
        gif_img_path = os.path.join(self.test_files_dir, 'gif_image_valid.gif')
        signature = self.scd_services.extract_features_signature_net1(gif_img_path)
        np.save(os.path.join(self.test_files_dir, 'gif_image_valid.npy'), signature)
        max_difference = np.max(np.abs(expected_output - signature))
        self.assertAlmostEqual(max_difference, 0.0, places=self.num_decimal_digits_accuracy)

    def test_SCD_extract_features_signature_net1_015(self) -> None:
        """
        SCD_extract_features_signature_net1_015: Extract signature from renamed .gif file (like renamed .jpeg or
        .bmp as .gif)
        """
        invalid_gif_file = os.path.join(self.test_files_dir, 'gif_image_invalid.gif')
        with self.assertRaises(ValueError) as err:
            self.scd_services.extract_features_signature_net1(invalid_gif_file)
        self.assertTrue((str(err.exception) == "Mismatch between actual image type and image name") != -1)

    def test_SCD_extract_features_signature_net1_016(self) -> None:
        """
        SCD_extract_features_signature_net1_016: Extract signature from .bmp image
        """
        expected_output = np.load(os.path.join(self.test_files_dir, 'bmp_image_valid.npy'))
        bmp_img_path = os.path.join(self.test_files_dir, 'bmp_image_valid.bmp')
        signature = self.scd_services.extract_features_signature_net1(bmp_img_path)
        np.save(os.path.join(self.test_files_dir, 'bmp_image_valid.npy'), signature)
        max_difference = np.max(np.abs(expected_output - signature))
        self.assertAlmostEqual(max_difference, 0.0, places=self.num_decimal_digits_accuracy)

    def test_SCD_extract_features_signature_net1_017(self) -> None:
        """
        SCD_extract_features_signature_net1_017: Extract signature from renamed .bmp file (like renamed .jpeg or
        .gif as .bmp)
        """
        invalid_bmp_file = os.path.join(self.test_files_dir, 'bmp_image_invalid.bmp')
        with self.assertRaises(ValueError) as err:
            self.scd_services.extract_features_signature_net1(invalid_bmp_file)
        self.assertTrue((str(err.exception) == "Mismatch between actual image type and image name") != -1)

    # # throughput test
    # def test_throughput(self) -> None:
    #     import time
    #
    #     start = time.perf_counter()
    #     jpg_img_path = os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid.JPG')
    #     for _ in range(100):
    #         self.scd_services.extract_features_signature_net1(jpg_img_path)
    #     end = time.perf_counter()
    #     print('average run time: {} sec'.format((end - start) / 100))
    #     self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
