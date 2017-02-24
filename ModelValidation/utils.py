import pandas as pd

def get_levels_stat(df, excl_fields=None):
    res = pd.DataFrame([], columns=["field", "level", "count", "perc"])
    for field in df.columns:
        if field in excl_fields:
            continue
        counts = df[field].value_counts(dropna=False)
        perc = 100 * counts/df.shape[0]
        levels_stat = pd.DataFrame({ "field": field, "level" : perc.index, "perc" : perc, "count": counts })
        res = res.append(levels_stat, ignore_index=True)
    return res