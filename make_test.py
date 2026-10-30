import pymupdf

text = """Tender Document
Eligibility: Bidder must have 5 years experience.
Delivery Schedule: 30 days from order date.
Payment Terms: 30 days after delivery.
Penalty: 1% per week for delay.
Performance Guarantee: 5% of contract value.
Warranty: 12 months.
Termination: Buyer may terminate with 30 days notice.
Arbitration: Disputes resolved by arbitration.
We will give this contract on sole source basis."""

doc = pymupdf.open()
page = doc.new_page()
page.insert_text((50, 72), text, fontsize=12)
doc.save("test_tender.pdf")
print("test_tender.pdf ready!")