prompt=f'''Based on the diagnose report provided, please summarize the report briefly and academically from the following perspectives as a medical professional, answer in phrases or medical vocabulary entity whenever possible to save words, don’t leave out important information. Connect the answers in one sequence, separated them with semicolons (important). Important notes: For all perspectives, focus on the microscopic description rather than gross description; If can’t answer from the specific perspective, just answer “Unknown.” without another words!!! 
1. What is the differentiation of the lesion? (e.g., Well-differentiated, Moderately differentiated, Poorly differentiated, Mixed differentiation, or other types.)
2. Are the margins of the excised tissue clear of disease? (Note: R0 indicates negative margins; R1 and R2 indicate positive margins; Rx indicates an unknown status, which should be recorded as ‘Unknown’.)
3. What is the histological classification or type of cancer, including DCIS if applicable?
4. What is the cancer subtype?
5. Is there a description of any necrosis?
6. Is there mention of tumor-infiltrating lymphocytes?
7. What is the histological grade?
8. What is the nuclear grade?
9. Is lymphovascular invasion present?
10. Is there any indication of calcification?
11. What is the receptor status?
12. What are the results of immunohistochemistry (IHC)?
report:{REPORT}'''