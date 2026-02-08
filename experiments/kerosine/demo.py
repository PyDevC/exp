import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import kero.Query.parser  as parser

query = """
WITH patient_vitals AS (
    SELECT 
        p.subject_id,
        p.gender,
        -- Calculating age at time of admission
        ROUND((CAST(a.admittime AS DATE) - CAST(p.dob AS DATE)) / 365.24, 2) AS age,
        ce.charttime,
        ce.valuenum AS heart_rate,
        a.hospital_expire_flag AS label -- 1 if the patient died, 0 otherwise
    FROM patients p
    INNER JOIN admissions a ON p.subject_id = a.subject_id
    INNER JOIN chartevents ce ON a.hadm_id = ce.hadm_id
    WHERE ce.itemid = 220045 -- Item ID for Heart Rate in Metavision
      AND ce.valuenum IS NOT NULL
      -- Focus on the first 24 hours of the stay for early prediction
      AND ce.charttime BETWEEN a.admittime AND (a.admittime + INTERVAL '1 day')
)
SELECT 
    subject_id,
    gender,
    age,
    -- Using window functions to order the sequence for the RNN
    ARRAY_AGG(heart_rate ORDER BY charttime) AS hr_sequence,
    ARRAY_AGG(EXTRACT(EPOCH FROM (charttime - MIN(charttime) OVER(PARTITION BY subject_id))) / 3600 ORDER BY charttime) AS time_deltas,
    label
FROM patient_vitals
GROUP BY subject_id, gender, age, label
ORDER BY subject_id;
"""

kero_parse = parser.Parser()
kero_parse.Capture(query)

class MimicDataset(Dataset):
    def __init__(self, sequences, static_features, labels):
        self.sequences = [torch.tensor(s, dtype=torch.float32).unsqueeze(-1) for s in sequences]
        self.static = torch.tensor(static_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.static[idx], self.labels[idx]

def collate_fn(batch):
    sequences, statics, labels = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_seqs, torch.stack(statics), torch.stack(labels)

class MortalityLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, static_dim):
        super(MortalityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + static_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, static):
        _, (hn, _) = self.lstm(x)
        last_hidden = hn[-1] 
        combined = torch.cat((last_hidden, static), dim=1)
        out = self.fc(combined)
        return self.sigmoid(out)

kero_source = kero.Source.connect(connection="postgresql://med_db")
loader = kero.data.DataLoader(
    kero_parse, 
    kero_source, 
    target_device="cuda:0",
    dtypes={"hr_sequence": torch.float32} 
)

# 5. Training Loop
model = MortalityLSTM(input_dim=1, hidden_dim=32, static_dim=2)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for seqs, statics, labels in loader:
        optimizer.zero_grad()
        outputs = model(seqs, statics).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
