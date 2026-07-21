from docx import Document

doc_path = "/Users/jasonandrews/Library/CloudStorage/Dropbox/Isaac-Jason/_Ebola/manuscript/LID Submission/R1/Lancet_ID_Reviewer_Responses.docx"
doc = Document(doc_path)

found = False
for i, para in enumerate(doc.paragraphs):
    if "Generate distributions of total degree, household/community degree" in para.text:
        para.text = "Response: We agree that these assumptions carry uncertainty. To address this, we have expanded the supplementary material to include the empirical basis and sources for these assumptions. Furthermore, we conducted a Probabilistic Sensitivity Analysis (PSA) to jointly propagate uncertainty in the primary epidemiological parameters. We used these uncertainty analyses for a sensitivity analysis by meta-regressing the parameters against the absolute vaccine impact from our simulations. We present the standardized beta coefficients from this multivariate linear regression to identify the key drivers of outbreak size. For the specific structural assumptions highlighted (four-day detection delay, household size, immune-onset period, and contact network characteristics), we performed detailed one-way sensitivity analyses, which are reported in the supplementary appendix, as generating novel synthetic networks for thousands of joint LHS samples is computationally prohibitive."
        # Make the "Response:" part bold
        para.runs[0].font.bold = True
        found = True
        break

if found:
    doc.save(doc_path)
    print("Successfully updated docx!")
else:
    print("Could not find the target paragraph.")
