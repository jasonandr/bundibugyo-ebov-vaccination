import pandas as pd
import numpy as np

def check():
    df = pd.read_csv('data_and_results/fig5_tornado_immune_results.csv')
    b = df[(df['scenario']=='analysis_1_reactive_ring') & (df['level']=='no_vax_enh_ops')][['seed','deaths_percent']]
    
    for level in ['vax_immune_5.0', 'vax_enh_ops', 'vax_immune_14.0']:
        i = df[(df['level']==level)][['seed','deaths_percent']]
        m = i.merge(b.rename(columns={'deaths_percent':'base_deaths'}), on='seed')
        v = np.where(m['base_deaths']>0, (m['base_deaths'] - m['deaths_percent'])/m['base_deaths']*100, np.nan)
        print(f"{level}: {np.nanmedian(v):.1f}%")

if __name__ == '__main__':
    check()
