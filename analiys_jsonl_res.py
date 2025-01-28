import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union

class JSONLProcessor:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def read_jsonl_files(self) -> List[Dict[str, Any]]:
        """Reads all JSONL files in the folder and returns a list of parsed records."""
        all_data = []
        files_in_folders = os.listdir(self.folder_path)
        print(files_in_folders)
        try:
            for file_name in files_in_folders:
                if file_name.endswith('.jsonl'):
                    file_path = os.path.join(self.folder_path, file_name)
                    try:
                        with open(file_path, 'r') as file:
                            all_data.extend([json.loads(line) for line in file])
                    except Exception as file_error:
                        print(f"Error reading file {file_name}: {file_error}")
        except Exception as folder_error:
            print(f"Error accessing folder {self.folder_path}: {folder_error}")
        return all_data

class ObjectExtractor:
    @staticmethod
    def convert_to_list(value: Any) -> List[float]:
        """Converts a string to a list if necessary."""
        if isinstance(value, str):
            try:
                value = value.replace("[", "").replace("]", "").split(",")
                return [float(cov) for cov in value]
            except ValueError:
                print(f"Failed to decode covariance: {value}")
                return []
        return value if isinstance(value, list) else []

    @staticmethod
    def safe_get(obj: Optional[Dict[str, Any]], key: str, default=None):
        """Safely retrieve a nested dictionary value."""
        if obj and isinstance(obj, dict):
            return obj.get(key, default)
        return default
    
    @staticmethod
    def get_success(obj: Union[bool, str, None]) -> bool:
        """Converts a success value to a boolean."""
        if obj is None:
            return False
        if isinstance(obj, bool):
            return obj
        return obj.lower() == 'true'

    # "Frame": {"Ego": {"TransformStamped": {"header": {"stamp": {"sec": 1606799233, "nanosec": 249851000}, "frame_id": "map"}, "child_frame_id": "base_link", "transform": {"translation": {"x": 89529.32151795654, "y": 42417.52324213837, "z": 6.100472927093506}, "rotation": {"x": -0.0012963204108964923, "y": -0.012038447733785842, "z": 0.8606085035719773, "w": 0.509123166737829}}, "rotation_euler": {"roll": 0.01940406194940797, "pitch": -0.014489861037339223, "yaw": 2.0733150481017883}}}
    @staticmethod
    def parse_ego(frame: Dict[str, Any]) -> Dict[str, Any]:
        """Parses ego information from the `Frame` field of each record."""
        ego = frame.get('Ego', {})
        transform = ego.get('TransformStamped', {})
        transform_data = transform.get('transform', {})
        header_stamp = transform.get('header', {}).get('stamp', {})
        if not header_stamp:
            return {}
        timestamp = header_stamp.get('sec', None) + header_stamp.get('nanosec', 0) * 1e-9
        translation = transform_data.get('translation', {})
        rotation = transform_data.get('rotation', {})

        return {
            'timestamp': timestamp,
            'object_type': 'Ego',
            'position_x': translation.get('x', None),
            'position_y': translation.get('y', None),
            'position_z': translation.get('z', None),
            'orientation_x': rotation.get('x', None),
            'orientation_y': rotation.get('y', None),
            'orientation_z': rotation.get('z', None),
            'orientation_w': rotation.get('w', None),
        }

    @staticmethod
    def extract_objects(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts object information from the `criteria0` field of each frame, including timestamps and covariance data."""
        extracted_objects = []

        for record in data:
            frame = record.get('Frame', {})
            if not frame:
                continue
            result = record.get('Result', {})
            stamp = record.get('Stamp', {}).get('ROS', None)
            success = result.get('Success', None)
            criteria0 = frame.get('criteria0', {})
            objects = criteria0.get('Objects', [])

            # append EGO info
            extracted_objects.append(ObjectExtractor.parse_ego(frame))

            # append object info
            for obj in objects:
                extracted_objects.append({
                    'timestamp': stamp,
                    'status': obj.get('status', 'Unknown'),
                    'object_type': obj.get('object_type', 'Unknown'),
                    'label': obj.get('label', ''),
                    'distance_from_ego': obj.get('distance_from_ego', None),
                    'position_x': ObjectExtractor.safe_get(obj.get('position'), 'x'),
                    'position_y': ObjectExtractor.safe_get(obj.get('position'), 'y'),
                    'position_z': ObjectExtractor.safe_get(obj.get('position'), 'z'),
                    'velocity_x': ObjectExtractor.safe_get(obj.get('velocity'), 'x'),
                    'velocity_y': ObjectExtractor.safe_get(obj.get('velocity'), 'y'),
                    'velocity_z': ObjectExtractor.safe_get(obj.get('velocity'), 'z'),
                    'orientation_x': ObjectExtractor.safe_get(obj.get('orientation'), 'x'),
                    'orientation_y': ObjectExtractor.safe_get(obj.get('orientation'), 'y'),
                    'orientation_z': ObjectExtractor.safe_get(obj.get('orientation'), 'z'),
                    'orientation_w': ObjectExtractor.safe_get(obj.get('orientation'), 'w'),
                    'pose_error_x': ObjectExtractor.safe_get(obj.get('pose_error'), 'x'),
                    'pose_error_y': ObjectExtractor.safe_get(obj.get('pose_error'), 'y'),
                    'pose_error_z': ObjectExtractor.safe_get(obj.get('pose_error'), 'z'),
                    'heading_error_z': ObjectExtractor.safe_get(obj.get('heading_error'), 'z'),
                    'velocity_error_x': ObjectExtractor.safe_get(obj.get('velocity_error'), 'x'),
                    'velocity_error_y': ObjectExtractor.safe_get(obj.get('velocity_error'), 'y'),
                    'bev_error': obj.get('bev_error', None),
                    'pose_covariance': ObjectExtractor.convert_to_list(obj.get('pose_covariance', [])),
                    'twist_covariance': ObjectExtractor.convert_to_list(obj.get('twist_covariance', [])),
                    'frame_success': ObjectExtractor.get_success(success)
                })

        return extracted_objects

class ObjectProcessor:
    def __init__(self, folder_path: str):
        self.processor = JSONLProcessor(folder_path)

    def process(self) -> pd.DataFrame:
        """Processes all JSONL files in the folder and returns a DataFrame of extracted objects."""
        data = self.processor.read_jsonl_files()
        objects = ObjectExtractor.extract_objects(data)
        return pd.DataFrame(objects)

if __name__ == "__main__":
    import sys
    # get argv from command line
    folder_name = "Nishishingjuku"
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
    # Path to the folder containing JSONL files
    this_folder = os.path.dirname(os.path.abspath(__file__))
    folder_path = this_folder + "/" + folder_name

    # Process the folder and extract objects
    processor = ObjectProcessor(folder_path)
    extracted_objects_df = processor.process()

    # Display or save the results
    print(extracted_objects_df.head())
    extracted_objects_df.to_csv("extracted_objects.csv", index=False)
