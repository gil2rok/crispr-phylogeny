import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    general = parser.add_argument_group("general output")
    general.add_argument("path")

    detailed = parser.add_argument_group("detailed output")
    detailed.add_argument("-l", "--long", action="store_true")
    
    args = parser.parse_args()
    
    # tree path
    # prior path
    # solvers to compare
    
    # if solver == 'heracles':
    #     num_epochs, embedding_dim, curvature, lr