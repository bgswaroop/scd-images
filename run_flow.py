from signature_net.sig_net_flow import SigNetFlow
from signature_net.utils import Utils
from similarity_net.sim_net_flow import SimNetFlow


def run_flow():
    """
    A high level function to run the train and evaluation flows
    :return: None
    """

    # Signature Net
    SigNetFlow.train()
    Utils.visualize_ae_input_output_pairs()
    Utils.save_avg_fourier_images()

    # Similarity Net
    SimNetFlow.train()


if __name__ == '__main__':
    try:
        run_flow()
    except Exception as e:
        print(e)
