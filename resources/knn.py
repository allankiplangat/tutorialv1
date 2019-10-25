from flask_restful import Resource, reqparse
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class Neighbours(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('observations',
                        type=bool,
                        required=True,
                        help="Obsevation is Required"
                        )

    parser.add_argument('n_neighbors',
                        type=int,
                        required=True,
                        help="Nearest Neighbours is Required"
                        )

    parser.add_argument('new_observation',
                        type=int,
                        required=True,
                        help="New observation is Required",
                        action='append'
                        )

    parser.add_argument('distance',
                        type=str,
                        required=True,
                        help="Distance is Required"
                        )

    def post(self):
        data = Neighbours.parser.parse_args()
        neigbour_number = data['n_neighbors']
        observed = data['new_observation']
        distance = data['distance']
        if data['observations']:
            # Load Data
            iris = datasets.load_iris()
            features = iris.data
            # Create standardizer
            standardizer = StandardScaler()
            # Standardize features
            features_standardized = standardizer.fit_transform(features)
            # Two nearest neighbors
            nearest_neighbors = NearestNeighbors(
                n_neighbors=neigbour_number, metric=distance).fit(features_standardized)
            # Create an observation
            new_observation = observed
            # Find distances and indices of the observation's nearest neighbors
            distances, indices = nearest_neighbors.kneighbors(
                [new_observation])
            matrix = nearest_neighbors.kneighbors_graph(features_standardized).toarray()
            for i, x in enumerate(matrix):
                x[i] = 0
            # View the nearest neighbors
            result = features_standardized[indices].tolist()
            return {
                'Nearest Neighbours': result,
                'Distances': distances.tolist(),
                'Matrix': matrix[0].tolist()
            }
