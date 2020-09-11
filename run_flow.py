from signature_net.sig_net_flow import SigNetFlow
from utils.visualization_utils import VisualizationUtils
from similarity_net.sim_net_flow import SimNetFlow

def run_flow():
    """
    A high level function to run the train and evaluation flows
    :return: None
    """
    SigNetFlow.train()
    # VisualizationUtils.visualize_ae_input_output_pairs()
    SimNetFlow.train()


if __name__ == '__main__':
    try:
        run_flow()
    except Exception as e:
        print(e)
