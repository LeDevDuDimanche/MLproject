import os
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from pipeline.ngrams_classif import NgramsExtractor
from pipeline.tsfresh_basic import TSFreshBasicExtractor
from pipeline.length_classif import LengthsExtractor

from utils.util import *
import time


def print_stats_paper(report,  output, avg='macro avg', stats=['mean', 'std']):
    by_label = report.groupby('label').describe()
    with open(output, "w") as f:
        for stat in stats:
            print >>f, "Statistic:", stat
            print >>f, by_label.loc[avg].xs(stat, level=1)


def print_boxplot(report, avg='macro avg',
                  cols=['precision', 'recall', 'f1-score']):
    report[report.label == 'macro avg'][cols].boxplot()
    plt.show()


def report_true_pred(y_true, y_pred, i, tag):
    tag += str(i)
    with open(tag, "w") as f:
        for i in range(0, len(y_true)):
            print >>f, y_true[i], y_pred[i]


def describe_classif_reports(results, tag):
    true_vectors, pred_vectors = [r[0] for r in results], [r[1] for r in results]
    all_folds = pd.DataFrame(columns=['label', 'fold', 'precision', 'recall', 'f1-score', 'support'])
    for i, (y_true, y_pred) in enumerate(zip(true_vectors, pred_vectors)):
        report_true_pred(y_true, y_pred, i, tag)
        output = classification_report(y_true, y_pred, output_dict=True)
        df = pd.DataFrame(output).transpose().reset_index().rename(columns={'index': 'label'})
        df['fold'] = i
        all_folds = all_folds.append(df)
    return all_folds


def cross_validate(df, output_acc, folds=10):
    kf = StratifiedKFold(n_splits=folds)
    results = []
    print df.groupby('class_label').count()
    for k, (train, test) in enumerate(kf.split(df, df.class_label)):
        print "Fold", k
        result = classify(df.iloc[train], df.iloc[test], output_acc)
        results.append(result)
    return results


def cross_classify(df1, df2, output_acc1, output_acc2, folds=10):
    df1_df2_accs, df2_df1_accs = [], []
    kf = StratifiedKFold(n_splits=folds)
    for k, ((df1_train, df1_test), (df2_train, df2_test)) in enumerate(zip(kf.split(df1, df1.class_label), kf.split(df2, df2.class_label))):
        print "Fold", k
        df1_df2_accs.append(classify(df1.iloc[df1_train], df2.iloc[df2_test], output_acc1))
        #df2_df1_accs.append(classify(df2.iloc[df2_train], df1.iloc[df1_test], output_acc2))
    return df1_df2_accs, df2_df1_accs

def classify(train, test, output_acc):
    # Feature extraction methods. Add/delete as required.
    combinedFeatures = FeatureUnion([
        ('tsfresh', TSFreshBasicExtractor()),
      ('ngrams', NgramsExtractor()),
    ])

    # Pipeline. Feature extraction + classification
    pipeline = Pipeline([
      ('features', combinedFeatures),
      ('clf', RandomForestClassifier(n_estimators=100))
    ])

    # Training with pipeline
    pipeline.fit(train, train.class_label)
    # Prediction
    y_pred = pipeline.predict(test)

    acc = accuracy_score(test.class_label, y_pred)
    print "Accuracy Score:", acc
    with open(output_acc, "a") as f:
        print >>f, "Accuracy Score:", acc
    return list(test.class_label), list(y_pred)


def remove_strange(df, shortlist, urls):

    strange_urls = []
    with open(shortlist) as f:
        lines = f.readlines()
        strange_urls = [x.strip() for x in lines]

    strange_urls = [urls.index(x) for x in strange_urls]
    print len(strange_urls)
    df = df[~df["class_label"].astype(int).isin(strange_urls)]
    return df


def get_interval(df, start_date, end_date):
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df = df.loc[mask]
    return df


def get_url_list(all_url_list):

    urls = []
    with open(all_url_list) as f:
        lines = f.readlines()
        urls = [x.strip() for x in lines]
    return urls

def normal_experiment():

    data_dir = '../data/'
    pickle_path = 'index.pickle'
    output_statistics = "results/stats" #precision/recall/f-score stats
    output_report = "results/report" #detailed report
    output_tp = "results/tp_" #true vs predicted label (for other analysis if required)
    output_acc = "results/acc" #accuracy for each fold
    all_url_list = "../short_list_500"

    urls = get_url_list(all_url_list)

    if os.path.isfile(pickle_path):
        df = load_data(path=pickle_path)
    else:
        df = load_data(path=data_dir)

    #opt_num_insts, opt_num_classes = optimal_instances_per_class(df, draw=True)
    #print "optimal", opt_num_insts, opt_num_classes
    num_samples = 60
    num_classes = 1500
    df_trimmed = trim_sample_df(df, num_samples, map(str, range(num_classes)))
    print "start cross validation"
    start = time.time()
    results = cross_validate(df_trimmed, output_acc)
    report = describe_classif_reports(results, output_tp)
    print_stats_paper(report, output_statistics)
    with open(output_report, "w") as f:
        f.write(report.to_string())
    stop = time.time()
    print "total time taken:", stop - start

if __name__ == '__main__':
    #Current classification pipeline using ngram features and random forests
    normal_experiment()
