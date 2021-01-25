import copy, kcm_clustering, math
import numpy as np, matplotlib.pyplot as plt
import kcm_clustering as kc

class fknn:
    def __init__(self, k, points, centers, clusters, m=2, num=10, min_val=0, max_val=1):
        self.num = num
        self.k = k
        self.centers = centers
        self.clusters = clusters
        self.m = m
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.org_points = copy.deepcopy(points)
        self.cover_points = []
        self.init_cover_points()
        
    
    # initializing new points to cover the coordinate plane    
    def init_cover_points(self):
        vals = np.mgrid[0:1:1/self.num, 0:1:1/self.num].reshape(2,-1).T
        for val in vals:
            new_point = kc.Point(val)
            self.cover_points.append(new_point)
        

    # plot the result
    def fknn_plot(self):
        colors = ["green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"]
        centers_x = [c[0] for c in self.centers]
        centers_y = [c[1] for c in self.centers]
        for i in range(len(self.clusters)):
            x_values = [y.values[0] for y in [x for x in self.clusters[i]]]
            y_values = [y.values[1] for y in [x for x in self.clusters[i]]]
            plt.scatter(x_values, y_values, c=colors[i])
        plt.scatter(centers_x, centers_y, c="red", s=40)
        plt.show()

    
    # calculating the distance between two points
    def calculate_distance(self, a, b):
        sum_square = 0
        for i in range(len(a)):
            sum_square += (a[i] - b[i])**2
        return math.sqrt(sum_square)

    
    # getting k nearest neighbors to a point
    def get_k_nearest(self, point):
        k_nearest_points = []
        k_nearest_distances_indices = []
        all_distances = []
        # calculating all distances
        for org_point in self.org_points:
            dist = self.calculate_distance(point.values, org_point.values)
            all_distances.append(dist)

        # finding k smallest distances indices
        k_nearest_distances_indices = np.argpartition(all_distances, self.k)[:self.k]

        # finding the corresponding points to k smallest distances
        k_nearest_distances = []
        for i in range(self.k):
            neighbor_point = copy.deepcopy(self.org_points[k_nearest_distances_indices[i]])
            k_nearest_points.append(neighbor_point)
            k_nearest_distances.append(all_distances[k_nearest_distances_indices[i]])
        return k_nearest_points, k_nearest_distances

    
    # calculating unit interval values for the given point using fknn algorithm
    def calculate_uis(self, point, k_nearest, distances):
        term1 = 0
        term2 = 0
        for i in range(len(self.centers)):
            for j in range(self.k):
                uij = k_nearest[j].unit_intervals[i]
                term1 += uij * ( 1 / (distances[j] ** (2/(self.m-1))))
                term2 += 1 / (distances[j] ** (2/(self.m-1)))
            ui = term1 / term2
            point.unit_intervals.append(ui)


    # label each cover point to show the cluster it belongs to
    def label_cover_points(self):
        for point in self.cover_points:
            max_index = 0
            for i in range(len(point.unit_intervals)):
                if point.unit_intervals[i] > point.unit_intervals[max_index]:
                    max_index = i
            point.label = max_index

    
    # add cover points to the cluster that they belong to
    def update_clusters(self):
        self.label_cover_points()
        for point in self.cover_points:
            self.clusters[point.label].append(point)            


    
    # running fknn algorithm
    def run_fknn(self):
        for point in self.cover_points:
            k_nearest, distances = self.get_k_nearest(point)
            self.calculate_uis(point, k_nearest, distances)
            self.update_clusters()



