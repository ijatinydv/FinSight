import sys
from pathlib import Path
from .config import cfg
from .data_loader import DataLoader

def main():
    loader = DataLoader()
    # For now just print counts of each dataset
    print(f"Profiles: {len(loader.get_profiles())}")
    print(f"Events: {len(loader.get_events())}")
    print(f"Requests: {len(loader.get_requests())}")
    # Future: run decision engine and write output.csv

if __name__ == "__main__":
    main()
