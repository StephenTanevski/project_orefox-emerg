from . import cleaner
from . import analyser
from . import plotter


def main():
    dc = cleaner.DataCleaner('test/data/BR20279856.csv', lab='ALS')
    dc.write_csv('results.csv')
    a = analyser.Analyser(dc)
    a.pca_sk()
    p = plotter.Plotter(a)
    p.plot_pca_feature_bar('filename.png')
    pass


if __name__ == '__main__':
    main()