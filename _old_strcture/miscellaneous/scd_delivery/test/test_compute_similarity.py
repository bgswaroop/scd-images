import os
import unittest
from collections import namedtuple

import numpy as np

from _old_strcture.miscellaneous.scd_delivery.main.scd_modules.services import ServicesSCD

ScoreRange = namedtuple('ScoreRange', 'min, max')


class TestComputeSimilarity(unittest.TestCase):

    def setUp(self) -> None:
        self.scd_services = ServicesSCD()
        self.scd_services.load_signature_net1()
        self.scd_services.load_similarity_net1()

        self.test_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unit_test_input_files')
        self.signature_size = self.scd_services.sig_size

    def tearDown(self) -> None:
        self.scd_services.unload_signature_net1()
        self.scd_services.unload_similarity_net1()

    def test_SCD_compute_correlation_similarity_net1_001(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_001: Compute the correlation between query signature and reference
        signature. There is  no correlation at  all or low one.
        """
        query_signature = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid_jpg.npy'))
        reference_signature = np.load(os.path.join(self.test_files_dir, 'FujiFilm_FinePixJ50_1_8184_valid_jpg.npy'))
        expected_score, expected_prediction = ScoreRange(min=0.0, max=0.5), False
        output_score, output_prediction = self.scd_services.compute_correlation_similarity_net1(query_signature,
                                                                                                reference_signature)
        self.assertEqual(expected_prediction, output_prediction, "Mismatch in prediction")
        self.assertGreaterEqual(output_score, expected_score.min,
                                f"Similarity score must be greater than {expected_score.min}")
        self.assertLessEqual(output_score, expected_score.max,
                             f"Similarity score must be less than {expected_score.max}")

    def test_SCD_compute_correlation_similarity_net1_002(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_002: Compute the correlation between query signature and reference
        signature. There is correlation.
        """
        query_signature = np.load(os.path.join(self.test_files_dir, 'Sony_DSC-T77_0_48385.npy'))
        reference_signature = np.load(os.path.join(self.test_files_dir, 'Sony_DSC-T77_0_48389.npy'))
        expected_score, expected_prediction = ScoreRange(min=0.5, max=1.0), True
        output_score, output_prediction = \
            self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature)
        self.assertEqual(expected_prediction, output_prediction, "Mismatch in prediction")
        self.assertGreaterEqual(output_score, expected_score.min,
                                f"Similarity score must be greater than {expected_score.min}")
        self.assertLessEqual(output_score, expected_score.max,
                             f"Similarity score must be less than {expected_score.max}")

    def test_SCD_compute_correlation_similarity_net1_003(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_003: Compute the correlation between query signature and reference
        signature. The threshold is manually specified. Score value less than threshold.
        """
        query_signature = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid_jpg.npy'))
        reference_signature = np.load(os.path.join(self.test_files_dir, 'FujiFilm_FinePixJ50_1_8184_valid_jpg.npy'))
        expected_score, expected_prediction = ScoreRange(min=0.0, max=0.95), False
        output_score, output_prediction = \
            self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature, 0.95)
        self.assertEqual(expected_prediction, output_prediction, "Mismatch in prediction")
        self.assertGreaterEqual(output_score, expected_score.min,
                                f"Similarity score must be greater than {expected_score.min}")
        self.assertLessEqual(output_score, expected_score.max,
                             f"Similarity score must be less than {expected_score.max}")

    def test_SCD_compute_correlation_similarity_net1_004(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_004: Compute the correlation between query signature and reference
        signature. The threshold is manually specified. Score value greater than threshold.
        """
        query_signature = np.load(os.path.join(self.test_files_dir, 'Sony_DSC-T77_0_48385.npy'))
        reference_signature = np.load(os.path.join(self.test_files_dir, 'Sony_DSC-T77_0_48389.npy'))
        expected_score, expected_prediction = ScoreRange(min=0.55, max=1.0), True
        output_score, output_prediction = \
            self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature, 0.55)
        self.assertEqual(expected_prediction, output_prediction, "Mismatch in prediction")
        self.assertGreaterEqual(output_score, expected_score.min,
                                f"Similarity score must be greater than {expected_score.min}")
        self.assertLessEqual(output_score, expected_score.max,
                             f"Similarity score must be less than {expected_score.max}")

    def test_SCD_compute_correlation_similarity_net1_005(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_005: Compute correlation between query signature and empty reference
        signature.
        """
        query_signature = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid_png.npy'))
        reference_signature = np.random.randn(0)
        with self.assertRaises(AssertionError) as err:
            self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature)
        self.assertTrue(str(err.exception) == "Reference signature size must be {}".format(self.signature_size))

    def test_SCD_compute_correlation_similarity_net1_006(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_006: Compute correlation between empty query signature and reference
        signature.
        """
        query_signature = np.random.randn(0)
        reference_signature = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid_png.npy'))
        with self.assertRaises(AssertionError) as err:
            self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature)
        self.assertTrue(str(err.exception) == "Query signature size must be {}".format(self.signature_size))

    def test_SCD_compute_correlation_similarity_net1_007(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_007: Compute correlation between empty query signature and empty
        reference signature.
        """
        query_signature = np.random.randn(0)
        reference_signature = np.random.randn(0)
        with self.assertRaises(AssertionError) as err:
            self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature)
        self.assertTrue(str(err.exception) == "Query signature size must be {}".format(self.signature_size))

    def test_SCD_compute_correlation_similarity_net1_008(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_008: Compute correlation between invalid query signature type and empty
        reference signature.
        """
        query_signature = [0] * self.signature_size  # list of int
        reference_signature = np.random.randn(0)
        with self.assertRaises(ValueError) as err:
            self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature)
        self.assertTrue(str(err.exception) == "Query signature must be of type numpy.ndarray")

    def test_SCD_compute_correlation_similarity_net1_009(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_009: Compute correlation between empty query signature and invalid type
        for reference signature.
        """
        query_signature = np.random.randn(0)
        reference_signature = ['0'] * self.signature_size  # list of str
        with self.assertRaises(ValueError) as err:
            self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature)
        self.assertTrue(str(err.exception) == "Reference signature must be of type numpy.ndarray")

    def test_SCD_compute_correlation_similarity_net1_010(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_010: Compute correlation between invalid query signature type and
        invalid type for reference signature.
        """
        query_signature = ['0'] * self.signature_size  # list of str
        reference_signature = [0] * self.signature_size  # list of int
        with self.assertRaises(ValueError) as err:
            self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature)
        self.assertTrue(str(err.exception) == "Query signature must be of type numpy.ndarray")

    def test_SCD_compute_correlation_similarity_net1_011(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_011: Compute correlation between invalid query signature type and valid
        reference signature.
        """
        query_signature = ['0'] * self.signature_size  # list of str
        reference_signature = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid_png.npy'))
        with self.assertRaises(ValueError) as err:
            self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature)
        self.assertTrue(str(err.exception) == "Query signature must be of type numpy.ndarray")

    def test_SCD_compute_correlation_similarity_net1_012(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_012: Compute correlation between valid query signature type and invalid
        reference signature.
        """
        query_signature = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_9_valid_png.npy'))
        reference_signature = ['0'] * self.signature_size  # list of str
        with self.assertRaises(ValueError) as err:
            self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature)
        self.assertTrue(str(err.exception) == "Reference signature must be of type numpy.ndarray")

    def test_SCD_compute_correlation_similarity_net1_013(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_013: Load and unload the model.
        """
        try:
            self.scd_services.load_similarity_net1()
            self.scd_services.unload_similarity_net1()
            self.setUp()
        except Exception as err:
            self.fail("Error in loading and unloading of similarity network")

    def test_SCD_compute_correlation_similarity_net1_014(self) -> None:
        """
        SCD_compute_correlation_similarity_net1_014: Test the method on 5 different inputs by consecutive evaluation.
        """
        # Initialize the placeholders
        num_tests = 5
        query_signature = [np.empty(0)] * num_tests
        reference_signature = [np.empty(0)] * num_tests
        expected_score = [0] * num_tests
        expected_prediction = [0] * num_tests
        output_score = [0] * num_tests
        output_prediction = [0] * num_tests

        # Prepare the inputs
        query_signature[0] = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid_jpg.npy'))
        reference_signature[0] = np.load(os.path.join(self.test_files_dir, 'FujiFilm_FinePixJ50_1_8184_valid_jpg.npy'))
        expected_score[0], expected_prediction[0] = 0.0, False

        query_signature[1] = np.load(os.path.join(self.test_files_dir, 'Sony_DSC-T77_0_48385.npy'))
        reference_signature[1] = np.load(os.path.join(self.test_files_dir, 'Sony_DSC-T77_0_48389.npy'))
        expected_score[1], expected_prediction[1] = 1.0, True

        query_signature[2] = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid_jpg.npy'))
        reference_signature[2] = np.load(os.path.join(self.test_files_dir, 'FujiFilm_FinePixJ50_1_8184_valid_jpg.npy'))
        expected_score[2], expected_prediction[2] = 0.0, False

        query_signature[3] = np.load(os.path.join(self.test_files_dir, 'Sony_DSC-T77_0_48389.npy'))
        reference_signature[3] = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid_jpg.npy'))
        expected_score[3], expected_prediction[3] = 0.0, False

        query_signature[4] = np.load(os.path.join(self.test_files_dir, 'FujiFilm_FinePixJ50_1_8184_valid_jpg.npy'))
        reference_signature[4] = np.load(os.path.join(self.test_files_dir, 'FujiFilm_FinePixJ50_1_8184_valid_jpg.npy'))
        expected_score[4], expected_prediction[4] = 1.0, True

        # Run the tests
        for i in range(num_tests):
            output_score[i], output_prediction[i] = \
                self.scd_services.compute_correlation_similarity_net1(query_signature[i], reference_signature[i])

        # Evaluate the results
        for i in range(num_tests):
            self.assertEqual(expected_prediction[i], output_prediction[i], "Mismatch in prediction")

    # # throughput test
    # def test_throughput(self) -> None:
    #     import time
    #
    #     start = time.perf_counter()
    #     query_signature = np.load(os.path.join(self.test_files_dir, 'Agfa_DC-504_0_25_valid_jpg.npy'))
    #     reference_signature = np.load(os.path.join(self.test_files_dir, 'FujiFilm_FinePixJ50_1_8184_valid_jpg.npy'))
    #     for _ in range(100):
    #         self.scd_services.compute_correlation_similarity_net1(query_signature, reference_signature)
    #     end = time.perf_counter()
    #     print('average run time: {} sec'.format((end - start) / 100))
    #     self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
