import torch
import torch.nn as nn
import kero

# 1. Define the Query (The "Schema" for your data)
query = """
SELECT 
    ARRAY_AGG(valuenum ORDER BY charttime) AS hr_sequence,
    gender, age, 
    hospital_expire_flag AS label
FROM mimic_data
GROUP BY subject_id, gender, age, hospital_expire_flag
"""

# 2. Setup Kero (Replacing Dataset, DataLoader, and Collate)
kero_parse = kero.Query.parser.Parser().Capture(query)
loader = kero.data.DataLoader(
    kero_parse, 
    source=kero.Source.connect("postgresql://med_db"),
    batch_size=32,
    dtypes={"hr_sequence": torch.float32}
)

# 3. Simple Model
class MortalityLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32 + 2, 1) # 32 hidden + (gender, age)

    def forward(self, x, static):
        _, (hn, _) = self.lstm(x.unsqueeze(-1))
        return torch.sigmoid(self.fc(torch.cat((hn[-1], static), dim=1)))

# 4. Minimal Training
model = MortalityLSTM()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(5):
    for seqs, statics, labels in loader:
        optimizer.zero_grad()
        loss = nn.BCELoss()(model(seqs, statics).squeeze(), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete.")
