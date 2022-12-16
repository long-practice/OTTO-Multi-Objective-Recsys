import os

import pandas as pd
import argparse

from prediction import predict


def load_submission_csv(train_path, test_path, model_path):
    session_interaction_dict = predict(train_path, test_path, model_path)
    submission_df = pd.DataFrame.from_dict(session_interaction_dict)

    submission_path = '../submission'
    submission_ls = os.listdir(submission_path)

    all_version = [int(file.split('.')[0][10:]) for file in submission_ls]
    if all_version:
        last_version = sorted(all_version)[-1]
    else:
        last_version = 0

    submission_df.load_csv(f'submission{last_version + 1}.csv')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path')
    parser.add_argument('--test_path')
    parser.add_argument('--model_path')
    args = parser.parse_args()

    load_submission_csv(train_path=args.train_path, test_path=args.test_path, model_path=args.model_path)