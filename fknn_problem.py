import kcm_clustering
import fknn_algorithm

min_clusters_num = 2
max_clusters_num = 10
m = 3
k = 10
num = 20


def init(file_name):
    points = []
    with open(file_name, 'r') as reader:
        for line in reader:
            values = [float(x) for x in line.split(',')]
            points.append(kcm_clustering.Point(values))
    return points


def main():
    # initializing points
    points = init("sample5.csv")

    # running KCM algorithm
    print("\nFirst of all we run KCM to determine the clusters (next step is FkNN)...\n")
    kcm = kcm_clustering.KCM(points, min_clusters_num, max_clusters_num, m)
    kcm.kcm_cluster()

    # determining the cluster borders using FkNN (fuzzy k-nearest neighbors) algorithm
    print("\nKCM finished. Running FkNN...\n")
    fknn = fknn_algorithm.fknn(k, kcm.points, kcm.centers, kcm.clusters, m, num)
    fknn.run_fknn()
    print("\nFkNN finished!")
    fknn.fknn_plot()


    # plotting the result
    # kcm.kcm_plot()




if __name__ == '__main__':
    main()