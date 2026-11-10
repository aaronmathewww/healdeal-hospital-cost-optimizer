"""
HealDeal — Synthetic Indian Hospital Dataset Generator
Generates synthetic demonstration records with simulated costs, attributes and coordinates.
Run once: python data/generate_dataset.py
"""
import pandas as pd
import numpy as np
import json, random, os
from pathlib import Path

random.seed(42)
np.random.seed(42)

CITIES = {
    "Delhi": (28.6139, 77.2090, "Delhi"),
    "Mumbai": (19.0760, 72.8777, "Maharashtra"),
    "Bengaluru": (12.9716, 77.5946, "Karnataka"),
    "Hyderabad": (17.3850, 78.4867, "Telangana"),
    "Chennai": (13.0827, 80.2707, "Tamil Nadu"),
    "Kolkata": (22.5726, 88.3639, "West Bengal"),
    "Pune": (18.5204, 73.8567, "Maharashtra"),
    "Ahmedabad": (23.0225, 72.5714, "Gujarat"),
    "Jaipur": (26.9124, 75.7873, "Rajasthan"),
    "Lucknow": (26.8467, 80.9462, "Uttar Pradesh"),
    "Chandigarh": (30.7333, 76.7794, "Chandigarh"),
    "Kochi": (9.9312, 76.2677, "Kerala"),
    "Bhopal": (23.2599, 77.4126, "Madhya Pradesh"),
    "Nagpur": (21.1458, 79.0882, "Maharashtra"),
    "Indore": (22.7196, 75.8577, "Madhya Pradesh"),
    "Surat": (21.1702, 72.8311, "Gujarat"),
    "Visakhapatnam": (17.6868, 83.2185, "Andhra Pradesh"),
    "Patna": (25.5941, 85.1376, "Bihar"),
    "Mysuru": (12.2958, 76.6397, "Karnataka"),
    "Coimbatore": (11.0168, 76.9558, "Tamil Nadu"),
}

SPECIALTIES = [
    "Cardiology", "Orthopedics", "Neurology", "Oncology",
    "Gastroenterology", "Nephrology", "Pulmonology", "Endocrinology",
    "Ophthalmology", "Dermatology", "General Surgery", "Pediatrics",
    "Obstetrics & Gynecology", "Urology", "Psychiatry", "ENT",
    "Rheumatology", "Hematology", "Plastic Surgery", "General Medicine",
]

PROCEDURES = {
    "Cardiology":    ("Angioplasty", 120000, 350000),
    "Orthopedics":   ("Knee Replacement", 80000, 280000),
    "Neurology":     ("Brain MRI + Consult", 15000, 60000),
    "Oncology":      ("Chemotherapy Cycle", 40000, 180000),
    "Gastroenterology": ("Endoscopy", 8000, 35000),
    "Nephrology":    ("Dialysis Session", 3000, 12000),
    "Pulmonology":   ("Bronchoscopy", 12000, 45000),
    "Endocrinology": ("Thyroid Panel", 2000, 8000),
    "Ophthalmology": ("Cataract Surgery", 15000, 70000),
    "Dermatology":   ("Laser Treatment", 5000, 30000),
    "General Surgery":("Appendectomy", 25000, 90000),
    "Pediatrics":    ("Vaccination Package", 1500, 6000),
    "Obstetrics & Gynecology": ("Normal Delivery", 30000, 120000),
    "Urology":       ("Kidney Stone Removal", 35000, 120000),
    "Psychiatry":    ("Therapy Session", 1000, 5000),
    "ENT":           ("Tonsillectomy", 20000, 65000),
    "Rheumatology":  ("Arthritis Panel", 4000, 15000),
    "Hematology":    ("Bone Marrow Biopsy", 18000, 60000),
    "Plastic Surgery":("Scar Revision", 15000, 80000),
    "General Medicine":("Full Checkup", 3000, 15000),
}

SAMPLE_REVIEWS = [
    "Excellent doctors and very clean facilities. Highly recommend!",
    "The staff was very caring and attentive throughout my stay.",
    "Good hospital but waiting time is too long. Doctors are knowledgeable.",
    "Affordable treatment with high quality care. Very satisfied.",
    "Modern equipment, experienced surgeons. Recovery was smooth.",
    "The nurses were kind but billing process was confusing.",
    "Best cardiac care in the city. Dr. was extremely professional.",
    "Average experience. Facilities could be better maintained.",
    "Very expensive but world class treatment. Worth every rupee.",
    "Quick diagnosis, proper medication. Would visit again.",
    "Hygiene standards are excellent. Staff is well trained.",
    "Had a long wait but the treatment quality was top notch.",
    "Doctors explain everything clearly. Very transparent billing.",
    "Emergency response was fast and efficient. Life saving care.",
    "Okay experience. Nothing exceptional but decent service.",
    "Overcrowded OPD but specialist consultation was great.",
    "State of the art infrastructure. Comfortable rooms.",
    "Friendly environment. My child felt comfortable during treatment.",
    "Expensive parking but hospital services are excellent.",
    "Not very happy with post-discharge follow up. Rest was fine.",
]

def generate_reviews(n=5):
    return "|".join(random.sample(SAMPLE_REVIEWS, min(n, len(SAMPLE_REVIEWS))))

rows = []
hospital_id = 1

for city, (base_lat, base_lon, state) in CITIES.items():
    n_hospitals = random.randint(25, 35)
    for _ in range(n_hospitals):
        spec = random.choice(SPECIALTIES)
        proc_name, cost_lo, cost_hi = PROCEDURES[spec]
        name = f"Demo Hospital H{hospital_id:04d}"
        tier = random.choice(["Premium", "Mid-Range", "Budget"])
        cost_mult = {"Premium": 1.4, "Mid-Range": 1.0, "Budget": 0.65}[tier]
        avg_cost = round(random.uniform(cost_lo, cost_hi) * cost_mult, -2)
        rating = round(np.clip(random.gauss(4.0, 0.6), 1.5, 5.0), 1)
        beds = random.choice([50, 100, 150, 200, 300, 500, 750, 1000])
        num_reviews = random.randint(30, 2000)
        sentiment = round(np.clip(random.gauss(0.65, 0.18), 0.0, 1.0), 3)
        lat = base_lat + random.uniform(-0.15, 0.15)
        lon = base_lon + random.uniform(-0.15, 0.15)
        accredited = random.choice([True, True, True, False])
        insurance = random.sample(
            ["Sample Insurance A", "Sample Insurance B", "Sample Insurance C",
             "Sample Insurance D", "Sample Insurance E"], k=random.randint(2, 5)
        )
        rows.append({
            "hospital_id": f"H{hospital_id:04d}",
            "name": name,
            "city": city,
            "state": state,
            "primary_specialty": spec,
            "primary_procedure": proc_name,
            "tier": tier,
            "avg_cost_for_procedure": avg_cost,
            "avg_rating": rating,
            "num_reviews": num_reviews,
            "sentiment_score": sentiment,
            "bed_count": beds,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "accredited": accredited,
            "insurance_accepted": "|".join(insurance),
            "reviews": generate_reviews(5),
        })
        hospital_id += 1

df = pd.DataFrame(rows)
out = Path(__file__).parent / "hospitals.csv"
df.to_csv(out, index=False)
print(f"✅ Generated {len(df)} hospitals → {out}")
print(df.groupby("city").size().to_string())
