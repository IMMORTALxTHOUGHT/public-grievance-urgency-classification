#!/usr/bin/env python3
"""
Data Acquisition Script
- Loads the Kaggle grievance dataset
- Maps CategoryV7 codes to top-level parent categories
- Maps to broad citizen-facing categories
- Samples and saves as CSV
"""
import json
import pandas as pd
import sys
from pathlib import Path

DATA_DIR = Path('/home/deep/.cache/kagglehub/datasets/ayushyajnik/government-of-india-grievance-report/versions/1')
OUTPUT_DIR = Path('/home/deep/projects/P4/public_grievance_urgency_classification_system/data')

def load_category_mapping():
    """Load and build a mapping from any category code to top-level parent."""
    df_map = pd.read_excel(DATA_DIR / 'CategoryCode_Mapping.xlsx')
    
    # Build parent lookup dict
    parent_lookup = {}
    desc_lookup = {}
    org_lookup = {}
    for _, row in df_map.iterrows():
        code = int(row['Code'])
        parent_lookup[code] = row['Parent']
        desc_lookup[code] = str(row['Description']).strip().replace('\n', ' ').replace('\r', '')
        org_lookup[code] = str(row['OrgCode']).strip()
    
    # Track top-level categories (Stage=1 = no parent)
    top_level = {}
    for _, row in df_map[df_map['Stage'] == 1].iterrows():
        code = int(row['Code'])
        top_level[code] = {
            'name': str(row['Description']).strip(),
            'org': str(row['OrgCode']).strip()
        }
    
    def find_top_parent(code):
        """Walk up the parent chain to find the top-level category."""
        visited = set()
        current = int(code)
        while current in parent_lookup and pd.notna(parent_lookup[current]):
            if current in visited:
                break
            visited.add(current)
            current = int(parent_lookup[current])
        return current
    
    # Pre-build mapping for all codes
    code_to_top = {}
    for code in df_map['Code']:
        code_int = int(code)
        top_code = find_top_parent(code_int)
        if top_code in top_level:
            code_to_top[code_int] = {
                'top_code': top_code,
                'top_name': top_level[top_code]['name'],
                'org': top_level[top_code]['org'],
                'sub_name': desc_lookup.get(code_int, 'Unknown'),
                'full_path': desc_lookup.get(code_int, 'Unknown')
            }
        else:
            # Fallback: use the code's own description
            code_to_top[code_int] = {
                'top_code': code_int,
                'top_name': desc_lookup.get(code_int, 'Unknown'),
                'org': org_lookup.get(code_int, ''),
                'sub_name': desc_lookup.get(code_int, 'Unknown'),
                'full_path': desc_lookup.get(code_int, 'Unknown')
            }
    
    return code_to_top, top_level

# Broad citizen-facing categories for our classification
# Map from the actual top_name values (which include "Ministry of" / "Department of" prefixes)
# to the broad citizen-facing categories our project expects
BROAD_CATEGORY_MAP = {
    # Road & Transport
    'Road Transport and Highways': 'Road & Transport',
    'Railway': 'Road & Transport',
    'Civil Aviation': 'Road & Transport',
    'Shipping': 'Road & Transport',
    # Healthcare
    'Health and Family Welfare': 'Healthcare',
    'Department of Health Research': 'Healthcare',
    'Ayush': 'Healthcare',
    # Water Supply
    'Ministry of Water Resources, River Development & Ganga Rejuv': 'Water Supply',
    'Ministry of Drinking Water and Sanitation': 'Water Supply',
    # Electricity / Power
    'Ministry of Power': 'Electricity',
    'Petroleum and Natural Gas': 'Electricity',
    'Ministry of New and Renewable Energy': 'Electricity',
    # Education
    'School Education': 'Education',
    'Higher Education': 'Education',
    'Ministry of Skill Development and Entrepreneurship': 'Education',
    # Public Safety
    'Home Affairs': 'Public Safety',
    'Personnel and Training': 'Public Safety',
    'Department of Justice': 'Public Safety',
    'Ministry of Electronics & Information Technology': 'Public Safety',
    'Labour and Employment': 'Public Safety',
    'Department of Social Justice and Empowerment': 'Public Safety',
    'Food and Public Distribution': 'Public Safety',
    'Department of Consumer Affairs': 'Public Safety',
    'Department of Defence': 'Public Safety',
    'Ministry of Corporate Affairs': 'Public Safety',
    # Housing
    'Housing and Urban Affairs': 'Housing',
    'Rural Development': 'Housing',
    # Sanitation / Urban Services (subset of Housing & Urban Affairs complaints)
    # Mapped separately via keyword-based reclassification in post-processing
    # Telecom / Technology
    'Telecommunications': 'Other',
    'Unique Identification Authority of India': 'Other',
    'Ministry of Information and Broadcasting': 'Other',
    # Banking / Finance / Insurance
    'Banking': 'Other',
    'Insurance': 'Other',
    'Department of Revenue': 'Other',
    'Department of Expenditure': 'Other',
    'Department of Economic Affairs ACC Division': 'Other',
    'Department of Financial Services (Pension Reforms)': 'Other',
    'Department of Investment & Public Asset Management': 'Other',
    # Other government services
    'Posts': 'Other',
    'Department of Agriculture and Farmers Welfare': 'Other',
    'Department of Agriculture Research and Education': 'Other',
    'External Affairs': 'Other',
    'Department for Promotion of Industry and Internal Trade': 'Other',
    'M/o Women and Child Development': 'Other',
    'Ministry of Environment, Forest and Climate Change': 'Other',
    'Ministry of Culture': 'Other',
    'Ministry of Textiles': 'Other',
    'Ministry of Micro Small and Medium  Enterprises': 'Other',
    'Ministry of Tourism': 'Other',
    'Ministry of Statistics and Programme Implementation': 'Other',
    'Department of Commerce': 'Other',
    'Ministry of Steel': 'Other',
    'Ministry of Mines': 'Other',
    'Ministry  of Coal': 'Other',
    'Department of Chemicals and Petrochemicals': 'Other',
    'Department of Fertilizers': 'Other',
    'Department of Heavy Industry': 'Other',
    'Department of Public Enterprises': 'Other',
    'Department of Land Resources': 'Other',
    'Department of Legal Affairs': 'Other',
    'Department of Empowerment of Persons with Disabilities': 'Other',
    'Department of Ex Servicemen Welfare': 'Other',
    'Department of Defence Finance': 'Other',
    'Department of Defence Production': 'Other',
    'Department of Defence Research and Development': 'Other',
    'Department of Military Affairs': 'Other',
    'Department of Atomic Energy': 'Other',
    'Department of Space': 'Other',
    'Department of Science and Technology': 'Other',
    'Department of Scientific & Industrial Research': 'Other',
    'Department of Bio Technology': 'Other',
    'NITI Aayog': 'Other',
    'Ministry of Panchayati Raj': 'Other',
    'Ministry of Parliamentary Affairs': 'Other',
    'Ministry of Cooperation': 'Other',
    'Ministry of Minority Affairs': 'Other',
    'Ministry of Tribal Affairs': 'Other',
    'Ministry of Development of North Eastern Region': 'Other',
    'Legislative Department': 'Other',
    'O/o the Comptroller & Auditor General of India': 'Other',
    'Department of Youth Affairs': 'Other',
    'Department of Sports': 'Other',
    'Department of Fisheries': 'Other',
    'Department of Pharmaceutical': 'Other',
    'Department of Official Language': 'Other',
    'Ministry of Earth Sciences': 'Other',
    'Central Board of Direct Taxes (Income Tax)': 'Other',
    'Central Board of Indirect Taxes and Customs': 'Other',
    'Department of Animal Husbandry, Dairying': 'Other',
    'Ministry of Food Processing Industries': 'Other',
    'Administrative Reforms and public grievances': 'Other',
}

def main():
    print("[1/4] Loading category mapping...")
    code_to_top, top_level = load_category_mapping()
    print(f"  Loaded {len(code_to_top)} category codes, {len(top_level)} top-level categories")
    
    print("[2/4] Loading grievance data...")
    with open(DATA_DIR / 'no_pii_grievance.json', 'r') as f:
        grievances = json.load(f)
    print(f"  Total records: {len(grievances)}")
    
    print("[3/4] Processing and mapping categories...")
    records = []
    unmapped = set()
    category_counts = {}
    
    for i, g in enumerate(grievances):
        if i % 20000 == 0 and i > 0:
            print(f"  Processed {i}/{len(grievances)}...")
        
        cat_code = g.get('CategoryV7')
        subject = g.get('subject_content_text', '')
        remarks = g.get('remarks_text', '')
        state = g.get('state', '')
        dist = g.get('dist_name', '')
        sex = g.get('sex', '')
        
        # Use subject + remarks as complaint text
        complaint_text = f"{subject} {remarks}".strip()
        if not complaint_text:
            continue
        
        # Map category
        if cat_code and int(cat_code) in code_to_top:
            top_info = code_to_top[int(cat_code)]
            top_name = top_info['top_name']
        else:
            unmapped.add(cat_code)
            top_name = 'Other'
        
        # Map to broad category
        broad_cat = BROAD_CATEGORY_MAP.get(top_name, 'Other')
        
        # Track counts for balanced sampling
        category_counts[broad_cat] = category_counts.get(broad_cat, 0) + 1
        
        records.append({
            'grievance_id': g.get('registration_no', f'GRIEV_{i}'),
            'complaint_text': complaint_text[:5000],  # cap length
            'category': broad_cat,
            'top_department': top_name,
            'state': state,
            'district': dist,
            'sex': sex,
            'org_code': g.get('org_code', ''),
        })
    
    print(f"  Records after filtering: {len(records)}")
    print(f"  Unmapped categories: {len(unmapped)}")
    
    print("\n  Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    
    df = pd.DataFrame(records)
    
    print("[4/4] Saving dataset...")
    # Create a balanced sample: cap "Other" and keep all others
    # This gives us a more ML-friendly distribution
    df_other = df[df['category'] == 'Other']
    df_non_other = df[df['category'] != 'Other']
    
    # Cap "Other" to maintain reasonable balance
    max_other = min(5000, len(df_other))
    df_other_sampled = df_other.sample(n=max_other, random_state=42)
    
    # Keep all non-Other records
    df_balanced = pd.concat([df_other_sampled, df_non_other], ignore_index=True)
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_path = OUTPUT_DIR / 'raw_grievances.csv'
    df_balanced.to_csv(output_path, index=False)
    print(f"  Saved {len(df_balanced)} records to {output_path}")
    print(f"  Category distribution:")
    for cat, count in sorted(df_balanced['category'].value_counts().items()):
        print(f"    {cat}: {count}")
    
    # Also save a smaller balanced sample for quick testing
    small_path = OUTPUT_DIR / 'sample_grievances.csv'
    df_small = df_balanced.groupby('category', group_keys=False).apply(
        lambda x: x.sample(n=min(50, len(x)), random_state=42)
    ).reset_index(drop=True)
    df_small.to_csv(small_path, index=False)
    print(f"  Saved {len(df_small)} balanced records to {small_path}")
    
    print("\nDone! Dataset ready.")

if __name__ == '__main__':
    main()
