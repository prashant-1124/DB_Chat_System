import os
import csv

file_path = r"D:\llm\powerbi_qa.csv"

print("File exists:", os.path.exists(file_path))

if os.path.exists(file_path):
    print("File size:", os.path.getsize(file_path), "bytes")

    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        print("CSV columns:", reader.fieldnames)

        rows = list(reader)
        print("Total CSV rows:", len(rows))

        for row in rows:
            if "incremental refresh" in str(row).lower():
                print("\nMATCH FOUND:")
                print(row)