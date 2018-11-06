import numpy as np
import pandas as pd
import pickle as pkl

import sklearn
import sys

class GradratesModel:

    def __init__(self, model_dir):
        with open(model_dir, 'rb') as f:
            self.model = pkl.load(f)

    def predict_gradrates(self, data, clean_and_augmnent=True):
        if clean_and_augmnent:
            df, df_id = self.clean_and_augment_data(data)
        return df, df_id, self.model.predict(df)

    def clean_and_augment_data(self,  df):
        #masks for gender and ethinicity
#        if df.GENDER == 'F':
#            female = True
#        else: female = False
        female = (df['GENDER'] == 'F')
        male = df['GENDER'] == 'M'
        ethnic_not_reported = df['ETHNIC'] == 0
        ethnic_asian = df['ETHNIC'] == 1
        ethnic_native = df['ETHNIC'] == 2
        ethnic_pacific_islander = df['ETHNIC'] == 3
        ethnic_filipino = df['ETHNIC'] == 4
        ethnic_hispanic = df['ETHNIC'] == 5
        ethnic_african_american = df['ETHNIC'] == 6
        ethnic_white = df['ETHNIC'] == 7
        ethnic_multi = df['ETHNIC'] == 9

        #Create new dataframes for gender and ethnic combinations, drop unused columns
        df_female_not = df[(female) & (ethnic_not_reported)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_male_not = df[(male) & (ethnic_not_reported)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_female_asi = df[(female) & (ethnic_asian)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_male_asi = df[(male) & (ethnic_asian)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_female_nat = df[(female) & (ethnic_native)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_male_nat = df[(male) & (ethnic_native)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_female_pac = df[(female) & (ethnic_pacific_islander)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_male_pac = df[(male) & (ethnic_pacific_islander)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_female_fil = df[(female) & (ethnic_filipino)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_male_fil = df[(male) & (ethnic_filipino)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_female_his = df[(female) & (ethnic_hispanic)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_male_his = df[(male) & (ethnic_hispanic)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])   
        df_female_afr = df[(female) & (ethnic_african_american)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_male_afr = df[(male) & (ethnic_african_american)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_female_whi = df[(female) & (ethnic_white)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_male_whi = df[(male) & (ethnic_white)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])     
        df_female_mul = df[(female) & (ethnic_multi)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])
        df_male_mul = df[(male) & (ethnic_multi)].drop(columns=['KDGN', 'GR_1', 'GR_2', 'GR_3', 'GR_4', 'GR_5', 'GR_6', 'GR_7', 'GR_8', 'UNGR_ELM', 'UNGR_SEC', 'ADULT'])  

        #Join separate dataframes back into one dataframe
        not_join = pd.merge(df_female_not, df_male_not, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer', suffixes=('_f_not', '_m_not'))
        asi_join = pd.merge(df_female_asi, df_male_asi, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer', suffixes=('_f_asi', '_m_asi'))
        nat_join = pd.merge(df_female_nat, df_male_nat, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer', suffixes=('_f_nat', '_m_nat'))
        pac_join = pd.merge(df_female_pac, df_male_pac, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer', suffixes=('_f_pac', '_m_pac'))
        fil_join = pd.merge(df_female_fil, df_male_fil, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer', suffixes=('_f_fil', '_m_fil'))
        his_join = pd.merge(df_female_his, df_male_his, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer', suffixes=('_f_his', '_m_his'))
        afr_join = pd.merge(df_female_afr, df_male_afr, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer', suffixes=('_f_afr', '_m_afr'))
        whi_join = pd.merge(df_female_whi, df_male_whi, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer', suffixes=('_f_whi', '_m_whi'))
        mul_join = pd.merge(df_female_mul, df_male_mul, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer', suffixes=('_f_mul', '_m_mul'))

        join_1 = pd.merge(not_join, asi_join, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer')   
        join_2 = pd.merge(join_1, nat_join, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer')
        join_3 = pd.merge(join_2, pac_join, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer')                                                            
        join_4 = pd.merge(join_3, fil_join, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer')
        join_5 = pd.merge(join_4, his_join, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer')                                                            
        join_6 = pd.merge(join_5, afr_join, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer')
        join_7 = pd.merge(join_6, whi_join, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer')
        join_8 = pd.merge(join_7, mul_join, on=['CDS_CODE', 'DISTRICT', 'COUNTY', 'SCHOOL'], how='outer')    
        
        df = join_8.fillna(value=0)
        #Enrollment totals
        df['enr_afr'] = df.ENR_TOTAL_f_afr+df.ENR_TOTAL_m_afr
        df['enr_asi'] = df.ENR_TOTAL_f_asi+df.ENR_TOTAL_m_asi
        df['enr_fil'] = df.ENR_TOTAL_f_fil+df.ENR_TOTAL_m_fil
        df['enr_his'] = df.ENR_TOTAL_f_his+df.ENR_TOTAL_m_his
        df['enr_mul'] = df.ENR_TOTAL_f_mul+df.ENR_TOTAL_m_mul
        df['enr_nat'] = df.ENR_TOTAL_f_nat+df.ENR_TOTAL_m_nat
        df['enr_not'] = df.ENR_TOTAL_f_not+df.ENR_TOTAL_m_not
        df['enr_pac'] = df.ENR_TOTAL_f_pac+df.ENR_TOTAL_m_pac
        df['enr_whi'] = df.ENR_TOTAL_f_whi+df.ENR_TOTAL_m_whi
        df['enr_tot'] = df.enr_afr+df.enr_asi+df.enr_fil+df.enr_his+df.enr_mul+df.enr_nat+df.enr_not+df.enr_pac+df.enr_whi

        #Enrollment ratios
        df['afr_f_ratio'] = df.ENR_TOTAL_f_afr/df.enr_tot
        df['afr_m_ratio'] = df.ENR_TOTAL_m_afr/df.enr_tot
        df['asi_f_ratio'] = df.ENR_TOTAL_f_asi/df.enr_tot
        df['asi_m_ratio'] = df.ENR_TOTAL_m_asi/df.enr_tot
        df['fil_f_ratio'] = df.ENR_TOTAL_f_fil/df.enr_tot
        df['fil_m_ratio'] = df.ENR_TOTAL_m_fil/df.enr_tot
        df['his_f_ratio'] = df.ENR_TOTAL_f_his/df.enr_tot
        df['his_m_ratio'] = df.ENR_TOTAL_m_his/df.enr_tot
        df['nat_f_ratio'] = df.ENR_TOTAL_f_nat/df.enr_tot
        df['nat_m_ratio'] = df.ENR_TOTAL_m_nat/df.enr_tot
        df['not_f_ratio'] = df.ENR_TOTAL_f_not/df.enr_tot
        df['not_m_ratio'] = df.ENR_TOTAL_m_not/df.enr_tot
        df['pac_f_ratio'] = df.ENR_TOTAL_f_pac/df.enr_tot
        df['pac_m_ratio'] = df.ENR_TOTAL_m_pac/df.enr_tot
        df['whi_f_ratio'] = df.ENR_TOTAL_f_whi/df.enr_tot
        df['whi_m_ratio'] = df.ENR_TOTAL_m_whi/df.enr_tot
        df['mul_f_ratio'] = df.ENR_TOTAL_f_mul/df.enr_tot
        df['mul_m_ratio'] = df.ENR_TOTAL_m_mul/df.enr_tot

        #Build high school mask       
        highschool_asi = (df['GR_9_f_asi']+df['GR_10_f_asi']+df['GR_11_f_asi']+df['GR_12_f_asi']+df['GR_9_m_asi']+df['GR_10_m_asi']+df['GR_11_m_asi']+df['GR_12_m_asi'])
        highschool_afr = (df['GR_9_f_afr']+df['GR_10_f_afr']+df['GR_11_f_afr']+df['GR_12_f_afr']+df['GR_9_m_afr']+df['GR_10_m_afr']+df['GR_11_m_afr']+df['GR_12_m_afr'])
        highschool_fil = (df['GR_9_f_fil']+df['GR_10_f_fil']+df['GR_11_f_fil']+df['GR_12_f_fil']+df['GR_9_m_fil']+df['GR_10_m_fil']+df['GR_11_m_fil']+df['GR_12_m_fil'])
        highschool_his = (df['GR_9_f_his']+df['GR_10_f_his']+df['GR_11_f_his']+df['GR_12_f_his']+df['GR_9_m_his']+df['GR_10_m_his']+df['GR_11_m_his']+df['GR_12_m_his'])
        highschool_nat = (df['GR_9_f_nat']+df['GR_10_f_nat']+df['GR_11_f_nat']+df['GR_12_f_nat']+df['GR_9_m_nat']+df['GR_10_m_nat']+df['GR_11_m_nat']+df['GR_12_m_nat'])
        highschool_not = (df['GR_9_f_not']+df['GR_10_f_not']+df['GR_11_f_not']+df['GR_12_f_not']+df['GR_9_m_not']+df['GR_10_m_not']+df['GR_11_m_not']+df['GR_12_m_not'])
        highschool_pac = (df['GR_9_f_pac']+df['GR_10_f_pac']+df['GR_11_f_pac']+df['GR_12_f_pac']+df['GR_9_m_pac']+df['GR_10_m_pac']+df['GR_11_m_pac']+df['GR_12_m_pac'])
        highschool_whi = (df['GR_9_f_whi']+df['GR_10_f_whi']+df['GR_11_f_whi']+df['GR_12_f_whi']+df['GR_9_m_whi']+df['GR_10_m_whi']+df['GR_11_m_whi']+df['GR_12_m_whi'])
        highschool_mul = (df['GR_9_f_mul']+df['GR_10_f_mul']+df['GR_11_f_mul']+df['GR_12_f_mul']+df['GR_9_m_mul']+df['GR_10_m_mul']+df['GR_11_m_mul']+df['GR_12_m_mul'])

        highschool_mask = (((highschool_asi+highschool_afr+highschool_fil+highschool_his+highschool_nat+highschool_not+highschool_pac+highschool_whi+highschool_mul) >0.0)&(df['SCHOOL'] !='District Total')&(df['SCHOOL'] !='State Total')&(df['SCHOOL'] !='County Total'))
        #Select only high schools
        df['highschool'] = (highschool_mask)
        df = df[df.highschool]
        df_model = df[['whi_f_ratio', 'whi_m_ratio', 'afr_f_ratio', 'afr_m_ratio', 'his_f_ratio', 'his_m_ratio', 'asi_f_ratio', 'asi_m_ratio', 'nat_f_ratio', 'nat_m_ratio', 'pac_f_ratio', 'pac_m_ratio', 'fil_f_ratio', 'fil_m_ratio', 'not_f_ratio', 'not_m_ratio', 'mul_f_ratio', 'mul_m_ratio']]
        df_model = df_model.fillna(value=0)
        df_id = df[['CDS_CODE', 'SCHOOL', 'whi_f_ratio', 'whi_m_ratio', 'afr_f_ratio', 'afr_m_ratio', 'his_f_ratio', 'his_m_ratio', 'asi_f_ratio', 'asi_m_ratio', 'nat_f_ratio', 'nat_m_ratio', 'pac_f_ratio', 'pac_m_ratio', 'fil_f_ratio', 'fil_m_ratio', 'not_f_ratio', 'not_m_ratio', 'mul_f_ratio', 'mul_m_ratio']]
            
        return df_model, df_id

def main(model_dir, data_dir, out_dir, data_type='table', clean_and_augmnent=True ):
    print('Starting')
    #Input dataframe
    if ( data_type == 'table' ):
        data = pd.read_table(data_dir)
    elif ( data_type == 'csv' ):
        data = pd.read_csv(data_dir)
    else: print( 'Data type not supported. Please use txt or csv file')

    #Initiate regression model    
    gradrates_model = GradratesModel(model_dir)

    #predict graduation rates
    df, df_id, pred = gradrates_model.predict_gradrates(data, clean_and_augmnent)

    #Add predicted graduation rates to cleaned dataframe that contains school code and name
    df_id['PRED_GRADRATE'] = pred

    #Store output
    df_id.to_csv(out_dir, index=False)
    print('Finished')

if __name__ == '__main__':
    main( *sys.argv[1:] )
