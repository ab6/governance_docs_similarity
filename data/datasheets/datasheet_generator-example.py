from governance_docs_similarity.datasheet_generator import generate_datasheet

datasheet = generate_datasheet(
    dataset_name="Loan Approval Training Data",
    version="1.0.0",
    description="Historical loan application outcomes from 2018-2023, "
                "used to train a binary classifier for loan pre-screening.",
    source="Internal loan management system, anonymized and aggregated",
    collection_method="Automated extraction from the loan processing database "
                      "with manual review of edge cases",
    size="50,000 applications (35,000 approved, 15,000 denied)",
    features=[
        {"name": "income", "type": "float", "description": "Annual income in USD"},
        {"name": "credit_score", "type": "int", "description": "FICO score (300-850)"},
        {"name": "debt_ratio", "type": "float", "description": "Total debt / annual income"},
    ],
    demographic_composition="Gender: 58% male, 42% female. Race: 64% white, "
        "18% Black, 12% Hispanic, 6% Asian. Age: median 38, range 21-72. "
        "Geographic: 70% urban, 30% rural.",
    known_biases="Historical approval rates show a 12% gap between male and "
        "female applicants with identical financial profiles. Black applicants "
        "have a 15% lower approval rate than white applicants at the same "
        "credit score tier. These disparities trace to historical lending "
        "practices. Applicant qualifications don't explain the gap.",
    preprocessing_steps=[
        "Removed applications with missing income or credit score (3.2% of records)",
        "Capped income at the 99th percentile to remove data entry errors",
        "Anonymized all personally identifiable information (name, SSN, address)",
        "Applied SMOTE oversampling to balance approval/denial ratio within each "
        "demographic group",
    ],
    intended_use="Pre-screening tool to flag applications likely to be denied, "
        "enabling early intervention by loan officers. Loan officers make the final decision.",
    prohibited_use="Must not be used as the sole basis for loan denial. Must not "
        "be deployed without the bias mitigation pipeline and human review queue.",
    retention_policy="Raw data retained for 7 years per federal banking regulations. "
        "Anonymized training set retained indefinitely.",
    contact="ml-governance@yourcompany.com",
)

with open("DATASHEET.md", "w") as f:
    f.write(datasheet)