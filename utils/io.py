from instance import TWDTSP

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path


class TWDTSPLoader:
    """
    Loader for TTP-format problem files.
    """

    DEFAULT_MIN_SPEED = 0.1
    DEFAULT_MAX_SPEED = 1.0
    DEFAULT_RENTING_RATIO = 0. # no time dependency
    DEFAULT_CAPACITY_OF_KNAPSACK = 0

    @staticmethod
    def load_from_file(filepath: str, populate: bool = False, db = None) -> 'TWDTSP':
        """
        Load a TWDTSP problem from a TTP-format file
        
        Args:
            filepath: Path to the problem file
            populate: If True, add problem metadata to the database
            db: Database connection/object to populate (required if populate=True)
        """
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        # header information
        metadata = {}
        coords = []
        items = {}
        
        section = None
        
        for line in lines:
            if not line or line.startswith('//'):
                continue
            
            # set section
            if 'NODE_COORD_SECTION' in line:
                section = 'coords'
                continue
            elif 'ITEMS SECTION' in line:
                section = 'items'
                continue
            
            # header section parsing
            if section is None and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    metadata[key] = value
            
            # coordinate section parsing
            elif section == 'coords':
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append([x, y])
                    except ValueError:
                        continue
            
            # items section parsing
            elif section == 'items':
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        item_idx = int(parts[0])
                        profit = float(parts[1])
                        weight = float(parts[2])
                        city = int(parts[3])
                        
                        if city not in items:
                            items[city] = []
                        items[city].append([profit, weight])
                    except ValueError:
                        continue
        
        # formatting
        coords_array = np.array(coords)
        items_dict = {city: np.array(city_items) for city, city_items in items.items()}        
        max_weight = float(metadata.get('CAPACITY OF KNAPSACK', TWDTSPLoader.DEFAULT_CAPACITY_OF_KNAPSACK))
        edge_weight_type = metadata.get('EDGE_WEIGHT_TYPE', 'CEIL_2D')        
        min_speed = float(metadata.get('MIN SPEED', TWDTSPLoader.DEFAULT_MIN_SPEED))
        max_speed = float(metadata.get('MAX SPEED', TWDTSPLoader.DEFAULT_MAX_SPEED))
        renting_ratio = float(metadata.get('RENTING RATIO', TWDTSPLoader.DEFAULT_RENTING_RATIO))
        
        problem = TWDTSP(coords_array, items_dict, max_weight, edge_weight_type)
        
        # Populate database if requested
        if populate:
            if db is None:
                raise ValueError("Database parameter 'db' must be provided when populate=True")
            
            problem_metadata = {
                'problem_name': metadata.get('PROBLEM NAME', 'unknown'),
                'min_speed': min_speed,
                'max_speed': max_speed,
                'renting_ratio': renting_ratio,
                'knapsack_data_type': metadata.get('KNAPSACK DATA TYPE', 'unknown'),
                'dimension': int(metadata.get('DIMENSION', len(coords))),
                'num_items': int(metadata.get('NUMBER OF ITEMS', sum(len(v) for v in items.values()))),
                'max_weight': max_weight,
                'edge_weight_type': edge_weight_type
            }
            
            TWDTSPLoader._populate_database(db, problem_metadata)
        
        return problem
    
    @staticmethod
    def _populate_database(db, metadata: Dict):
        """
        Populate the database with problem metadata.
        
        This is a placeholder method that should be customized based on your database type.
        Examples for different database types:
        
        # For pandas DataFrame:
        # db.loc[len(db)] = metadata
        
        # For SQLite:
        # cursor = db.cursor()
        # columns = ', '.join(metadata.keys())
        # placeholders = ', '.join(['?' for _ in metadata])
        # cursor.execute(f"INSERT INTO problems ({columns}) VALUES ({placeholders})", 
        #                tuple(metadata.values()))
        # db.commit()
        
        # For SQL with pandas:
        # pd.DataFrame([metadata]).to_sql('problems', db, if_exists='append', index=False)
        """
        # Generic implementation - attempts to add as a row
        if hasattr(db, 'loc'):  # pandas DataFrame
            db.loc[len(db)] = metadata
        elif hasattr(db, 'cursor'):  # SQL connection
            cursor = db.cursor()
            columns = ', '.join(metadata.keys())
            placeholders = ', '.join(['?' for _ in metadata])
            cursor.execute(f"INSERT INTO problems ({columns}) VALUES ({placeholders})", 
                          tuple(metadata.values()))
            db.commit()
        else:
            raise NotImplementedError(
                f"Database type {type(db).__name__} not supported. "
                "Please customize _populate_database method for your database type."
            )
    
    @staticmethod
    def load_multiple(filepaths: List[str], populate: bool = False, db = None) -> List['TWDTSP']:
        """
        Load multiple problem instances from a list of files
        
        Args:
            filepaths: List of file paths to load
            populate: If True, add problem metadata to the database
            db: Database connection/object to populate (required if populate=True)
        """
        problems = []
        for filepath in filepaths:
            try:
                problem = TWDTSPLoader.load_from_file(filepath, populate=populate, db=db)
                print(f"✓ Loaded: {filepath} ({problem.n} cities)")
            except Exception as e:
                print(f"✗ Failed to load {filepath}: {e}")
        
        return problems
    
    @staticmethod
    def load_from_directory(directory: str, pattern: str = "*.txt", populate: bool = False, db = None) -> List['TWDTSP']:
        """
        Load all matching problem files from a directory
        
        Args:
            directory: Directory path containing problem files
            pattern: Glob pattern for matching files
            populate: If True, add problem metadata to the database
            db: Database connection/object to populate (required if populate=True)
        """
        dir_path = Path(directory)
        filepaths = [str(f) for f in dir_path.glob(pattern)]
        return TWDTSPLoader.load_multiple(filepaths, populate=populate, db=db)


# Example usage
if __name__ == "__main__":    
    # Create a database (using pandas DataFrame as example)
    db = pd.DataFrame(columns=[
        'problem_name', 'min_speed', 'max_speed', 'renting_ratio',
        'knapsack_data_type', 'dimension', 'num_items', 'max_weight', 'edge_weight_type'
    ])
    
    # Load a single file without populating database
    problem = TWDTSPLoader.load_from_file("./data/a280_n279_bounded-strongly-corr_01.ttp")
    print(f"Problem loaded: {problem.n} cities, {problem.W} max weight")
    
    # Load a single file and populate database
    problem = TWDTSPLoader.load_from_file(
        "./data/a280_n279_bounded-strongly-corr_01.ttp",
        populate=True,
        db=db
    )
    
    # Load multiple files and populate database
    problems = TWDTSPLoader.load_multiple([
        "./data/a280_n279_bounded-strongly-corr_01.ttp",
        "./data/a280_n1395_uncorr-similar-weights_05.ttp",
    ], populate=True, db=db)
    
    print(f"\nLoaded {len(problems)} problems")
    print("\nDatabase contents:")
    print(db)
    
    # Load all files from a directory and populate database
    all_problems = TWDTSPLoader.load_from_directory(
        "./data", 
        pattern="*ttp",
        populate=True,
        db=db
    )
    print(f"\nTotal problems in database: {len(db)}")