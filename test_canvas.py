import pandas as pd
import matplotlib.pyplot as plt

class DatasetManager:
    def __init__(self, file_path, category_mapping):
        """
        Initializes the DatasetManager with the data file and category mapping.

        Parameters:
            file_path (str): Path to the CSV file.
            category_mapping (dict): Mapping of labels to categories.
        """
        self.data = pd.read_csv(file_path)
        self.category_mapping = category_mapping
        self._apply_category_mapping()

    def _apply_category_mapping(self):
        """Applies the category mapping to the dataset."""
        self.data['general_category'] = self.data['label'].apply(
            lambda x: self.map_to_category(x)
        )

    def map_to_category(self, label):
        """Maps a label to a predefined category or marks it as 'unclassified'."""
        reverse_mapping = {}
        for category, labels in self.category_mapping.items():
            for item in labels:
                reverse_mapping[item] = category
        return reverse_mapping.get(label, 'unclassified')

    def calculate_category_metrics(self, category):
        """Calculates metrics (TPrate and mAP) for a specific category."""
        category_data = self.data[self.data['general_category'] == category]
        tp = len(category_data[category_data['status'] == 'TP']) / 2  # Adjust for double counting
        fp = len(category_data[category_data['status'] == 'FP'])
        fn = len(category_data[category_data['status'] == 'FN'])

        tprate = tp / (tp + fn) if (tp + fn) > 0 else 0
        map_value = tp / (tp + fp) if (tp + fp) > 0 else 0

        return {'TPrate': tprate, 'mAP': map_value}

    def calculate_metrics(self):
        """Calculates TPrate and mAP for each category in the dataset."""
        metrics = {}
        categories = self.data['general_category'].unique()
        for category in categories:
            if category != 'unclassified':
                metrics[category] = self.calculate_category_metrics(category)
        return metrics

    def distance_based_fp_tp_fn(self, bin_size=10):
        """
        Calculates FP, TP, and FN counts based on distance intervals.

        Parameters:
            bin_size (int): Size of each distance bin (e.g., 10m).

        Returns:
            DataFrame: Aggregated FP, TP, and FN counts by distance bins.
        """
        self.data['distance_bin'] = (self.data['distance_from_ego'] // bin_size) * bin_size
        grouped = self.data.groupby(['distance_bin', 'status']).size().unstack(fill_value=0)
        return grouped

    def visualize_distance_fp_tp_fn(self, bin_size=10):
        """
        Visualizes FP, TP, and FN counts based on distance intervals.

        Parameters:
            bin_size (int): Size of each distance bin (e.g., 10m).
        """
        grouped = self.distance_based_fp_tp_fn(bin_size)
        grouped.plot(kind='bar', stacked=True, figsize=(12, 8))
        plt.title('FP/TP/FN Counts by Distance Intervals')
        plt.xlabel('Distance from Ego Vehicle (m)')
        plt.ylabel('Count')
        plt.legend(title='Status')
        plt.tight_layout()
        plt.grid()
        plt.show()

    def distance_based_categoly_analysis(self, bin_size=10):
        """
        Calculates object counts by distance intervals and categories.

        Parameters:
            bin_size (int): Size of each distance bin (e.g., 10m).

        Returns:
            DataFrame: Aggregated object counts by distance bins and categories.
        """
        self.data['distance_bin'] = (self.data['distance_from_ego'] // bin_size) * bin_size
        grouped = self.data.groupby(['distance_bin', 'general_category']).size().unstack(fill_value=0)
        return grouped

    def visualize_distance_analysis_by_category(self, bin_size=10):
        """
        Visualizes object counts by distance intervals and categories.

        Parameters:
            bin_size (int): Size of each distance bin (e.g., 10m).
        """
        grouped = self.distance_based_categoly_analysis(bin_size)
        grouped.plot(kind='bar', stacked=True, figsize=(12, 8))
        plt.title('Object Counts by Distance Intervals and Categories')
        plt.xlabel('Distance from Ego Vehicle (m)')
        plt.ylabel('Object Count')
        plt.legend(title='Category')
        plt.tight_layout()
        plt.grid()
        plt.show()

def main():
    """Main function to process the data and calculate metrics."""
    file_path = 'extracted_objects.csv'
    category_mapping = {
        'car': ['car', 'vehicle.car'],
        'large_vehicle': ['bus', 'vehicle.bus (bendy & rigid)', 'truck', 'trailer', 'vehicle.truck', 'vehicle.trailer', 'vehicle.construction', 'vehicle.emergency (ambulance & police)'],
        'pedestrian': ['pedestrian', 'pedestrian.adult', 'pedestrian.child', 'pedestrian.stroller'],
        'bike': ['bicycle', 'vehicle.bicycle', 'vehicle.motorcycle'],
    }

    manager = DatasetManager(file_path, category_mapping)

    # Calculate and print metrics
    metrics = manager.calculate_metrics()
    print("Metrics:")
    for category, metric in metrics.items():
        print(f"{category}: TPrate={metric['TPrate']:.2f}, mAP={metric['mAP']:.2f}")

    # Visualize distance-based FP/TP/FN analysis
    manager.visualize_distance_fp_tp_fn(bin_size=10)

    # Visualize distance-based analysis by category
    manager.visualize_distance_analysis_by_category(bin_size=10)

if __name__ == "__main__":
    main()
