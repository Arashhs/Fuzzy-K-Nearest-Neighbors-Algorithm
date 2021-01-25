import kcm_clustering
import fknn

min_clusters_num = 2
max_clusters_num = 10
m = 2


def init(file_name):
    points = []
    with open(file_name, 'r') as reader:
        for line in reader:
            values = [float(x) for x in line.split(',')]
            points.append(kcm_clustering.Point(values))
    return points


def main():
    # initializing points
    points = init("test3.csv")

    # running KCM algorithm
    kcm = kcm_clustering.KCM(points, min_clusters_num, max_clusters_num, m)
    kcm.kcm_cluster()

    # determining the cluster borders using FkNN (fuzzy k-nearest neighbors) algorithm


    # plotting the result
    kcm.kcm_plot()




if __name__ == '__main__':
    main()