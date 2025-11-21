from .helper_functions import describe_numeric_col

def outliers(data, cont_vars, out_summ_loc):
    cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                                 upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv(out_summ_loc)
    print(outlier_summary)