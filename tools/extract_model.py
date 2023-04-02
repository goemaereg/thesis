import torch
import argparse

def extract_model(path_in, path_out):
    print(f'{path_in} -> {path_out}')
    state_dict = torch.load(path_in, map_location=torch.device('cpu'))['model_state_dict']
    torch.save(state_dict, path_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='saved model state file')
    parser.add_argument('--output', '-o', type=str, help='target model file')
    args = parser.parse_args()
    extract_model(args.input, args.output)
