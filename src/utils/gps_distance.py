"""
This is a small script to calculate the distance between the cores of the case study and all other cores in the dataset.
It has problem when executed under the current folder, which is caused by logging.py.
But I currently have no intention to modiy the name of logging.py, which requres a cautious check.
Hence, if you want to run this script, please copy it to the root folder.
"""

import pandas as pd
import geopy.distance as gd


def get_distance(case_cores):
    gps_df = pd.read_excel("data/legacy/ML station list.xlsx")
    gps_df.columns = ['station', 'latitude', 'longitude', 'water depth (m)']

    for core in case_cores:
        row_case = gps_df[gps_df["station"] == core].squeeze()
        corrd_case = (row_case["latitude"],
                      row_case["longitude"])
        print(f"Distance to {core}:")

        for _, row in gps_df.iterrows():
            coord1 = (row["latitude"], row["longitude"])
            print(gd.distance(corrd_case, coord1).km)

        print("="*10)


if __name__ == "__main__":
    get_distance(["PS75-056-1", "LV28-44-3", "SO264-69-2"])
