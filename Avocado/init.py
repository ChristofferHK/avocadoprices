import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

'''

    Innledende program for å lage visualiseringer for å forstå dataen
    vi skulle jobbe med.
    Selve maskinlæringen ble utført i Azure Machine Learning Studio

'''

frame = pd.read_csv('avocado.csv')

df = frame.copy()

df['Date'] = pd.to_datetime(df['Date'])

print(df.head())
print(df['AveragePrice'].describe())
# print(df.tail(10))
# print(f"Shape of df: {df.shape}\n")


# print(df['AveragePrice'].describe())
# print(df.apply(pd.Series.value_counts))

# Viser verdien på gjennomsnittspris over hele perioden
def dist_plot(dataset, plot):
    plot.figure(figsize=(12, 5))
    plot.title("Distribution Price")

    ax = sns.distplot(dataset['AveragePrice'], color='r')
    plot.show()


# Gir veldig lite verdi. Ingen tydelig korrelasjon på pris
def corr_matrix(dataset, plot, sb):
    corr = dataset.corr()
    # corr.style.background_gradient(cmap="coolwarm").set_precision(2)
    sb.set(style="dark")
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sb.diverging_palette(240, 9, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, square=True,
                cbar_kws={"shrink": .5})
    plot.show()


# Ikke bra, altfor mye shit på grafen
def strip_plot(dataset, plot, sb):
    sns.set(style="darkgrid")
    data = dataset[['AveragePrice', 'region', 'year']]
    melted_df = pd.melt(data, value_vars=["AveragePrice", "region"])
    print(melted_df.shape)
    f, ax = plot.subplots()
    sns.despine(bottom=True, left=True)

    sb.stripplot(x="AveragePrice", y="region", hue="year",
                 data=data, palette="magma")

    plot.show()


def cat_plot(dataset, plot, sb):
    sns.set(style="ticks")
    data = dataset[['AveragePrice', 'region', 'year']]

    g = sns.catplot(x="AveragePrice", y="region", hue="year",
                    data=data, kind="point", join=False,
                    palette="deep", height=14, aspect=0.7)
    plot.show()


# cat_plot(df, plt, sns)


def latest(dataset, plot, sb):
    sns.set(style="ticks")

    # filtrerer ut sett fra dataen hvor året ikke er 2017/2018
    temp = dataset[dataset.year.isin([2017, 2018])]
    data = temp[['AveragePrice', 'region', 'year']]

    box = sns.boxplot(x='region', y='AveragePrice', hue='year',
                      data=data, palette="RdBu")
    
    box.set_xticklabels(box.get_xticklabels(), rotation=45)

    plot.show()


def price_history_based_on_type(dataset, plot, sb):
    sns.set(style="darkgrid")

    data = dataset[['Date', 'AveragePrice', 'type']]

    sb.lineplot(x="Date", y="AveragePrice", hue="type",
                data=data)
    plot.show()


#cat_plot(df, plt, sns)
#price_history_based_on_type(df, plt, sns)
latest(df, plt, sns)
