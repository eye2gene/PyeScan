# Grouby patient number
# and then group by another value and accregate predictions
def get_results_from_dataframe(df, classes=None, group_on='file_path', patient_id_key='patient_number'):
    import numpy as np
    
    if classes is None:
        classes = [ cls.replace("pred_","") for cls in df.columns
                    if cls.startswith("pred_")
                    and not cls in ["pred_class", "pred_model"] ]

    pandas_query = [ 'pred_'+cls for cls in classes ]
    
    truth, pred_scores, pat_ids = list(), list(), list()
    for pat_id, pat_entries in df.groupby(patient_id_key):
        gene = pat_entries['gene'].values[0]
        gene_ind = classes.index(gene)
        for val, grouped_entries in pat_entries.groupby(group_on):
            pred_scores.append(grouped_entries[pandas_query].mean(axis=0))
            pat_ids.append(pat_id)
            truth.append(gene_ind)   
    return np.array(truth), np.array(pred_scores), pat_ids, classes