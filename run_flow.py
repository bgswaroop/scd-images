from signature_net.sig_net_flow import SigNetFlow


def run_flow():
    """
    A high level function to run the train and evaluation flows
    :return: None
    """
    SigNetFlow.train()


if __name__ == '__main__':
    run_flow()
