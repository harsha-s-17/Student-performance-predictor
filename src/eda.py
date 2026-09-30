import matplotlib.pyplot as plt
import seaborn as sns


def score_distribution_figure(data):
    fig, ax = plt.subplots()
    sns.histplot(data["G3"], bins=11, kde=True, ax=ax)
    ax.set(title="Final grade distribution", xlabel="Final grade (G3)", ylabel="Students")
    return fig


def studytime_figure(data):
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x="studytime", y="G3", ax=ax)
    ax.set(title="Study time vs final grade", xlabel="Study time group", ylabel="Final grade (G3)")
    return fig


def numeric_relationship_figure(data, x, y, title):
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x=x, y=y, ax=ax)
    ax.set(title=title)
    return fig


def categorical_average_figure(data, x, y, title):
    fig, ax = plt.subplots()
    data.groupby(x, as_index=False)[y].mean().plot.bar(x=x, y=y, ax=ax, legend=False)
    ax.set(title=title, xlabel=x, ylabel=f"Average {y}")
    return fig


def correlation_figure(data):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data.select_dtypes(include="number").corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric feature correlation heatmap")
    return fig
