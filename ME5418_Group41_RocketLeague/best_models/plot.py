import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt


def plotter(data_dict, keyword):
    data_interest = {}
    for key, value in data_dict.items():
        if keyword in key:
            data_interest[key] = value

    cols = ["mean_reward_100", "goal_rate_100"]
    for metric in cols:
        plt.figure(figsize=(16,12))
        for key, value in data_interest.items():
            x = data_dict[key]["step"]
            y = data_dict[key][metric]
            plt.plot(x, y, label=key)
        title = f"{metric} - {list(data_interest.keys())}"
        plt.title(title)
        plt.xlabel("steps")
        plt.ylabel(metric)
        plt.legend(loc="upper left")
        plt.savefig(f"./{title}")


def extract_data():
    csv_files = glob.glob("./**/*.csv")
    data_dict = {}
    for file in csv_files:
        data_dict[file.split("/")[1]] = pd.read_csv(file)
    return data_dict


def main(args=None):
    keywords = ["none", "heuristic", "self_r1", "self_r2"]
    data_dict = extract_data()
    for keyword in keywords:
        plotter(data_dict, keyword)

if __name__ == "__main__":
    main(sys.argv)