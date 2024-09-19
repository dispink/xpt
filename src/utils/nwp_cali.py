import pandas as pd
import numpy as np


class PrepareData:
    """
    This is a class gathering the functions to prepare data. Be careful,
    the using of measurement (CaCO3%, TC%, TOC%) data need to follow the 
    selected measurment in the initialization because the filtering is
    based that selection only. 

    Example:
        prepare = PrepareData(measurement='CaCO3%')
        X, y = prepare.produce_Xy(prepare.select_data())
    """

    def __init__(self, measurement,
                 data_dir='~/CaCO3_NWP/data/spe+bulk_dataset_20220629.csv',
                 select_dir='~/CaCO3_NWP/data/ML station list.xlsx',
                 channel_amount=2048):
        while True:
            if measurement not in ['CaCO3%', 'TC%', 'TOC%']:
                print('The measurement should be strings of CaCO3%, TC% or TOC%.')
                break
            else:
                self.measurement = measurement
            self.data_dir = data_dir
            self.select_dir = select_dir
            self.channel_amount = channel_amount
            break

    def select_data(self):
        """
        This function is to select the chosen cores in the file 
        (select_dir) and having the measurement (CaCO3, TC, TOC) from a 
        pd.DataFrame (data_dir). This dataset has composite_id, 
        channels, TC%, TOC%, CaCO3%, core, mid_depth_mm and the output
        will be in the same format. The negative measurement values are
        excluded.
        """
        data_df = pd.read_csv(self.data_dir, index_col=0)
        xl_df = pd.read_excel(self.select_dir, sheet_name='CHOSEN')
        mask = ((data_df.core.isin(xl_df.Station)) &
                (~data_df[self.measurement].isna()) &
                (data_df[self.measurement] >= 0))
        return data_df.loc[mask, :]

    def select_casestudy(self, case_cores=['SO202-37-2_re', 'PS75-056-1']):
        """
        This function is to select the two cores for case study and 
        having the measurement (CaCO3, TC, TOC) from a pd.DataFrame 
        (data_dir). This dataset has composite_id, channels, TC%, TOC%,
        CaCO3%, core, mid_depth_mm and the output will be in the same 
        format. The negative measurement values are excluded.

        You can choose to take other cores as the casestudy cores.
        """
        data_df = pd.read_csv(self.data_dir, index_col=0)
        mask = ((data_df.core.isin(case_cores)) &
                (~data_df[self.measurement].isna()) &
                (data_df[self.measurement] >= 0))
        return data_df.loc[mask, :]

    def produce_Xy(self, data_df):
        """
        This function takes the input pd.DataFrame, usually the output 
        of self.select_data (the data shouldn't have NAs and negative 
        value), to generate X (channels normalized by the row sum) and
        y (weight percent, the 0 values are replaced by 0.01). The 
        ouputs are in np.ndarray
        """
        X = data_df.iloc[:, :-5].values
        X = X / X.sum(axis=1, keepdims=True)
        y = data_df[self.measurement].replace(0, 0.01).values

        return X, y


class Quantify:
    """
    This is a class for applying our model to quantify CaCO3 and TOC
    conveniently. The directories are in relative path because this
    way will be easier for others to apply. We assume others will 
    just fork the whole repository to whereever they want to store.
    We don't use absolute path also because we won't do these work on
    the HPC.

    The class doesn't deal with the depth or other infomation. Only
    the input spectra and output wt% are considered. The alignment of
    those information will be needed.

    The models are in the latest released version. The confidence
    intervals (ci) are adopted from build_model_11.ipynb.
    """

    def __init__(self, measurement,
                 model_dir={
                     'CaCO3%': 'released_models/caco3_nmf+svr_model_20210823.joblib',
                     'TOC%': 'released_models/toc_nmf+svr_model_20210823.joblib'},
                 ci={
                     'CaCO3%': [6.766615478157537, 5.910238821180132],
                     'TOC%': [0.5405406125751289, 0.3040019840141711]},
                 channel_amount=2048):
        while True:
            if measurement not in ['CaCO3%', 'TOC%']:
                print(
                    'The measurement should be strings of CaCO3% or TOC%.')
                break
            else:
                from joblib import load
                self.measurement = measurement
                self.model = load(model_dir[measurement])
                self.channel_amount = channel_amount
                self.ci = ci[measurement]
            break

    def ScaleX(self, data_df):
        """
        This function takes the input pd.DataFrame having spectra as
        the first 2048 columns (the data shouldn't have NAs and 
        negative value), to generate X (channels normalized by the 
        row sum). The ouputs are in np.ndarray
        """
        X = data_df.iloc[:, :self.channel_amount].values
        return X / X.sum(axis=1, keepdims=True)

    def predict(self, X):
        """
        This function feeds the scaled spectra np.ndarray to the 
        selected model. The model gives quantification in ln space so
        the output is then exponentialized to normal space, which is
        in wt%. 
        """
        y = self.model.predict(X)
        return np.exp(y)

    def print_CI(self):
        print('Lower cutoff, Upper cutoff')
        print(self.ci)


if __name__ == '__main__':
    prepare = PrepareData(measurement='CaCO3%')
    data_df = prepare.select_casestudy(case_cores=['PS75-056-1'])
    quan = Quantify(measurement='CaCO3%')
    X = quan.ScaleX(data_df)
    print(quan.predict(X))
