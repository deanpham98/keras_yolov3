import os
import sys
import pdb

import numpy as np 
import pandas as pd 


# def calc_dividend_value(row):
# 	try:
# 		return row['Dividend Value'] + dividend_info[row['Date']]

# def calc_adjusted_dividend():


# def calc_reinvested_dividend():


# def calc_total_value():




if __name__ == '__main__':
	price_records = pd.read_excel('price_records_1.xlsx').set_index('Date')
	price_records = price_records.iloc[::-1]

	dividend_info = pd.read_excel('dividend_info_1.xlsx')
	dividend_info = dividend_info.drop(labels=['Distribution Type', 'Currency', 'Income Dividend Taxation'], axis=1)
	dividend_info = dividend_info.iloc[::-1]
	dividend_info['Ex-Date'] = dividend_info['Ex-Date'].apply(lambda x: pd.Timestamp.strptime(x, '%d-%b-%Y'))
	dividend_info['Payment Date'] = dividend_info['Payment Date'].apply(lambda x: pd.Timestamp.strptime(x, '%d-%b-%Y') if x == x else x)
	dividend_info = dividend_info[(dividend_info['Payment Date'].isnull() | ((dividend_info['Payment Date'] < price_records.index[-1]) & (dividend_info['Ex-Date'] > price_records.index[0])))]
	
	price_records['Dividend Value'] = np.nan
	null_info = dividend_info[dividend_info['Payment Date'].isnull()]
	price_records['Dividend Value'].loc[null_info['Ex-Date']] = null_info['Value'].tolist()
	pdb.set_trace()
	complete_info = dividend_info[dividend_info['Payment Date'].notnull()]
	price_records['Dividend Value'].loc[complete_info['Payment Date']] = complete_info['Value'].tolist()


	fill_info = dividend_info.bfill(axis=0).loc[null_info.index]
	fill_info = fill_info.groupby(['Payment Date'])['Value'].sum()
	price_records['Dividend Value'].loc[fill_info.index] += fill_info.tolist()

	print('price_records: ', price_records[price_records['Dividend Value'].notnull()])
