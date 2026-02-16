from torch.utils.data import Dataset, DataLoader
import sqlite3
import pandas as pd

class SQLDataset(Dataset):
    def __init__(self, db_path, query):
        self.conn = sqlite3.connect(db_path)
        # Load data once or stream it
        self.df = pd.read_sql_query(query, self.conn)
        self.data = torch.tensor(self.df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return features and a dummy target for training
        sample = self.data[idx]
        return sample[:-1], sample[-1]

# Usage
# dataset = SQLDataset('local_warehouse.db', "SELECT * FROM training_table")
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
