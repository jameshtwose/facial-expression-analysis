# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from jmspack.NLTSA import (
    ts_levels,
    fluctuation_intensity,
    distribution_uniformity,
    complexity_resonance,
    complexity_resonance_diagram,
)
from jmspack.utils import JmsColors, apply_scaling

# %%
if "jms_style_sheet" in plt.style.available:
    plt.style.use("jms_style_sheet")

# %%
pre_df = pd.read_csv("data/face_expressions.csv")
pre_df.columns = [x.replace(" ", "") for x in pre_df.columns.tolist()]
AU_list = pre_df.filter(regex="AU[\d\D][\d\D]_r").columns.tolist()
time_column = "timestamp"
df = pre_df[[time_column] + AU_list]
# %%
df.head()
# %%
df.info()
# %%
df.columns.tolist()
# %%
df.describe()
# %%
_ = sns.heatmap(df.set_index(time_column).corr(), cmap="coolwarm")
# %%
_ = sns.heatmap(df.set_index(time_column).pipe(apply_scaling).T)
# %%
plot_df = df.melt(id_vars=time_column)
_ = plt.figure(figsize=(20, 10))
_ = sns.lineplot(data=plot_df, x=time_column, y="value", hue="variable")
# %%
ts_df, fig, ax = ts_levels(ts=df[AU_list[2]], figsize=(10, 5), plot=False)
# %%
ts_df
# %%
fi_df = fluctuation_intensity(df=df[AU_list].pipe(apply_scaling),
                      win=60,
                      xmin=0,
                      xmax=1,
                      col_first=1,
                      col_last=df[AU_list].shape[1],)
# du_df = distribution_uniformity(df=df[AU_list].pipe(apply_scaling),
#                         win=60,
#                         xmin=0,
#                         xmax=1,
#                         col_first=1,
#                         col_last=df[AU_list].shape[1],)
# cr_df = complexity_resonance(distribution_uniformity_df=du_df,
#                         fluctuation_intensity_df=fi_df)
# %%
fig, ax = complexity_resonance_diagram(df=fi_df, figsize=(10, 5))
# %%
