import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def load_data():
    return torch.FloatTensor(pd.read_csv("./data/digit-recognizer/test.csv").values)

def test(model, test_X):
    model.eval()
    pred = []
    with torch.no_grad():
        _test_X = test_X.split(64, dim=0)
        for x in tqdm(_test_X):
            pred.append(model(x))
    return torch.cat(pred, dim=0)
      
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_X = load_data().to(device).reshape(-1, 28, 28)
    model_fn = './model_files/model_DNN.pth'
    model = torch.load(model_fn)

    pred = test(model, test_X)

    pred_df = pd.DataFrame({
        "ImageId": np.arange(len(pred)) + 1,
        "Label": torch.argmax(pred, dim=-1).cpu().numpy()
    })
    pred_df.to_csv('./submissions/DNN.csv', index=False)
    print("done.")

if __name__ == '__main__':
    main()
