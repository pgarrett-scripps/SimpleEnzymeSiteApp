from collections import Counter
from io import StringIO
from typing import Tuple, Dict, Literal

import numpy as np
import peptacular.sequence
import streamlit as st
import pandas as pd
from filterframes import from_dta_select_filter
from constants import COMPARISON_AA_FREQUENCIES
from plotly import express as px

def get_enzyme_sites(peptides: list[str]) -> Tuple[Counter[Tuple[str, str]],Counter[Tuple[str, str]]]:
    n_term_sites, c_term_sites = Counter(), Counter()
    for peptide in peptides:
        aa_before = peptide[0]
        aa_after = peptide[-1]

        unmod_peptide = peptacular.sequence.strip_mods(peptacular.sequence.convert_ip2_sequence(peptide))
        n_term_site = (aa_before, unmod_peptide[0])
        n_term_sites[n_term_site] += 1

        c_term_site = (unmod_peptide[-1], aa_after)
        c_term_sites[c_term_site] += 1

    return n_term_sites, c_term_sites

@st.cache_data()
def get_dta_select_counts(files: list[st.file_uploader], sites: Literal['n', 'c', 'both']) -> Counter[Tuple[str, str]]:
    peptides = []
    for file in files:
        file_io = StringIO(file.getvalue().decode("utf-8"))
        _, peptide_df, _, _ = from_dta_select_filter(file_io)
        peptides.extend([peptide for peptide in peptide_df['Sequence'].tolist()])

    if sites == 'n':
        n_term_sites, _ = get_enzyme_sites(peptides)
        return n_term_sites
    elif sites == 'c':
        _, c_term_sites = get_enzyme_sites(peptides)
        return c_term_sites
    elif sites == 'both':
        n_term_sites, c_term_sites = get_enzyme_sites(peptides)
        return n_term_sites + c_term_sites
    else:
        raise ValueError(f'Invalid sites value: {sites}')


def calculate_baseline_site_freqs(freqs: dict[str, float]) -> Dict[Tuple[str, str], float]:
    aa_freqs = {}
    for aa1 in freqs:
        for aa2 in freqs:
            aa_freqs[(aa1, aa2)] = freqs[aa1] * freqs[aa2]

    return aa_freqs


def calculate_log2fold_change(observed_freqs: Dict[Tuple[str, str], float],
                              baseline_freqs: Dict[Tuple[str, str], float]):
    all_sites = set(observed_freqs.keys()).union(baseline_freqs.keys())
    log2fold_changes = {}
    for site in all_sites:
        observed_freq = observed_freqs.get(site, 0)
        baseline_freq = baseline_freqs.get(site, 0)
        if baseline_freq == 0:
            log2fold_changes[site] = np.inf
        else:
            log2fold_changes[site] = np.log2(observed_freq / baseline_freq)

    return log2fold_changes


with st.sidebar:
    st.title("Enzyme Site Explorer")

    filter_files = st.file_uploader("Upload DtaSelect-filter.txt files", accept_multiple_files=True, type='.txt')
    baseline_aa_freq = COMPARISON_AA_FREQUENCIES[st.selectbox('Select baseline AA frequency', COMPARISON_AA_FREQUENCIES.keys())]
    sites_to_compare = st.selectbox('Select sites to compare', ['n', 'c', 'both'], index=2)

    run = st.button('Run', use_container_width=True, type='primary')

st.subheader('Enzyme Site Frequencies Comparison')

with st.expander('Help'):
    st.markdown("""
    
    This tool compares the frequency of observed cleavage sites compared to the frequency of the expected cleavage
    sites based on random chance, which is calculated from the product of the frequency of the two amino acids 
    comprising the cleavage site. 
    
    The tool provides 3 options for sites to compare:
    
    - N-Terminus: Compares the frequency of the amino acid before the peptide to the first amino acid in the peptide
    - C-Terminus: Compares the frequency of the last amino acid in the peptide to the amino acid after the peptide
    - Both: Compares both the N-Terminus and C-Terminus sites
    
    The above options can be important to consider when analyzing enzyme cleavage sites. Some residues 
    at the end of a peptide can improve/hurt the likelihood of detection which is be secondary to the enzymes activity. 
    Since enzymatic cleavage occurs before, ionization, fragmentation, and detection, the observed enzymatic sites are
    influenced by all the the later events. Even if a site is highly cleave-able, if the actual peptide
    cannot be detected, the site will not be observed and the frequency will be lower than expected (or visa versa).
    
    Information on Columns:
    
    - Log2Fold Change: The log2 fold change of the observed frequency compared to the baseline frequency
    - Observed Frequency: The frequency of the observed site
    - Baseline Frequency: The frequency of the baseline site
    - Observed Count: The number of observed sites
    - Expected Count: The number of expected sites
    - Normalized Observed Frequency: The observed frequency normalized to the maximum observed frequency
    - Normalized Baseline Frequency: The baseline frequency normalized to the maximum baseline frequency
    
    """)

if run:


    if not filter_files:
        st.write('No files uploaded')
        st.stop()

    site_counts = get_dta_select_counts(filter_files, sites_to_compare)

    baseline_site_freq = calculate_baseline_site_freqs(baseline_aa_freq)

    # add all aa's from baseline to aa_freqs
    for site in baseline_site_freq:
        if site not in site_counts:
            site_counts[site] = 0

    sites_to_remove = set()
    for site in site_counts:
        if site not in baseline_site_freq:
            sites_to_remove.add(site)

    for site in sites_to_remove:
        del site_counts[site]

    total_sites = sum(site_counts.values())
    site_freqs = {site: count / total_sites for site, count in site_counts.items()}

    log2fold_changes = calculate_log2fold_change(site_freqs, baseline_site_freq)

    df = pd.DataFrame.from_dict(log2fold_changes, orient='index', columns=['Log2Fold Change'])
    df['Observed Frequency'] = df.index.map(site_freqs.get)
    df['Baseline Frequency'] = df.index.map(baseline_site_freq.get)
    df['Observed Count'] = df.index.map(site_counts.get)
    df['Expected Count'] = df['Baseline Frequency'] * total_sites
    df['Expected Count'] = df['Expected Count'].astype(int)

    df['Observed Frequency'] = df['Observed Frequency'] * 100
    df['Baseline Frequency'] = df['Baseline Frequency'] * 100
    df['Normalized Observed Frequency'] = df['Observed Frequency'] / max(df['Observed Frequency'])
    df['Normalized Baseline Frequency'] = df['Baseline Frequency'] / max(df['Baseline Frequency'])

    # sort df by index (aa1, aa2)
    df = df.sort_index()


    # fill inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)


    # Apply a gradient color map to the Log2Fold Change column
    styled_df = df.style.background_gradient(subset=['Log2Fold Change'], cmap='coolwarm')

    # Display the styled dataframe in Streamlit
    st.dataframe(styled_df, column_config={
        'Log2Fold Change': st.column_config.NumberColumn(format="%.2f"),
    }, use_container_width=True)
    #st.subheader('AA Frequency Log2Fold Change')
    #st.bar_chart(df['Log2Fold Change'], x_label='Amino Acid', y_label='Log2Fold Change')

    df[['AA1', 'AA2']] = pd.DataFrame(df.index.tolist(), index=df.index)



    # Create a heatmap of the log2fold changes
    @st.fragment()
    def plot_funct():
        col_options = df.columns.tolist()
        col_options.remove('AA1')
        col_options.remove('AA2')
        selected_column = st.selectbox('Select the column to plot:', col_options)

        # Add a dropdown menu for column selection

        # Pivot the dataframe to reshape it for the heatmap based on the selected column
        heatmap_data = df.pivot(index='AA2', columns='AA1', values=selected_column)

        # Create a heatmap using plotly with updated title and labels
        heatmap = px.imshow(heatmap_data,
                            labels={'color': selected_column, 'x': 'AA1', 'y': 'AA2'},
                            color_continuous_scale='RdBu_r',
                            height=800, width=800,  # Set plot size
                            title=f'{selected_column} Heatmap',  # Update title dynamically
                            text_auto='.1f')  # Format text to 1 decimal place

        # Display the heatmap in Streamlit
        st.plotly_chart(heatmap)


    plot_funct()



