import re

def parse_receipt(text):
    data = {}
    data['total'] = re.search(r'Total:\s*\$?(\d+\.\d{2})', text, re.IGNORECASE)
    data['date'] = re.search(r'(\d{2}/\d{2}/\d{4})', text)
    data['store_name'] = re.search(r'(Store|Shop|Merchant):\s*(.*)', text, re.IGNORECASE)

    # Extract matched groups
    for key, match in data.items():
        data[key] = match.group(1) if match else "Not Found"
    return data
