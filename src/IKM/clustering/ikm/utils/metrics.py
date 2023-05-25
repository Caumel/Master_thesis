import scipy.special
import math
import pandas as pd
from sklearn import metrics


class Metrics:

    def purity(self, df_main, label):

        """
        Calculates clusters purity.
        """

        cps = []
        cluster_values = df_main['Cluster'].unique().tolist()
        for i in cluster_values:
            if i == -1:
                continue
            df = df_main[df_main['Cluster'] == i].copy()
            frequent_value = df[label].value_counts().idxmax()
            num_frequent_value_cluster = df[label].value_counts().max()
            num_frequent_value_overall = (df_main[label].values == frequent_value).sum()
            cp = num_frequent_value_cluster / num_frequent_value_overall
            cps.append(cp)

        if not cps:
            return 0
        return (sum(cps) / len(cps)) * 100

    def rand_index(self, df, label):

        """
        Calculates Rand index.
        """

        labels_true = df[label].tolist()
        labels_true_float = [float(x) for x in labels_true]
        labels_pred = df['Cluster'].tolist()
        rand_index = metrics.rand_score(labels_true_float, labels_pred)
        return rand_index

    def information_criterion(self, df, label):

        """
        Calculates information criterion.
        """

        contingency_matrix = pd.crosstab(index=df[label], columns=df['Cluster'], margins=True)
        classes_total = df[label].nunique()
        clusters_total = df['Cluster'].nunique()

        n_objects = len(df.index)
        tmp = 0

        for k in range(0, clusters_total):
            for c in range(0, classes_total):
                tmp += (contingency_matrix.iloc[c, k] / n_objects) * math.log10(
                    (contingency_matrix.iloc[c, k] / contingency_matrix.iloc[classes_total, k]) + 0.000000000000001)
        emp_cond_entropy = -tmp

        n_binom = []
        k_binom = classes_total - 1
        for i in range(0, clusters_total):
            n_binom.append(contingency_matrix.iloc[classes_total, k] + classes_total - 1)

        numb_bits_encode_cont_matrix = 0
        for i in range(clusters_total):
            numb_bits_encode_cont_matrix += math.log10((scipy.special.binom(n_binom[i], k_binom)) + 0.000000000000001)

        information_criterion = emp_cond_entropy + numb_bits_encode_cont_matrix / n_objects

        return information_criterion
