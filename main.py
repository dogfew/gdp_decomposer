import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from decomposer import Decomposer


def main():

    df = pd.read_csv('gdp.csv', sep=';')
    branches = ['Yp', 'Cp', 'Gp', 'Jp', 'Exp', 'Imp']
    decomposer = Decomposer(df, branches, n_components=3)
    decomposer.fit(multistarts=10, reg=0.1, x0=None)
    decomposer.fit(multistarts=1, reg=0, x0=decomposer, verbose=False)
    ax = df[branches].plot(linestyle='--')
    sns.lineplot(x=df.t, y=decomposer.first_component, label='pA', linestyle='-',
                 linewidth=2, ax=ax)
    sns.lineplot(x=df.t, y=decomposer.second_component, label='pB', linestyle='-',
                 linewidth=2, ax=ax)
    if len(decomposer.components) > 2:
        sns.lineplot(x=df.t, y=decomposer.third_component, label='pC', linestyle='-', linewidth=2, ax=ax)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Time")
    plt.show()
    print(decomposer.summary())


if __name__ == '__main__':
    main()