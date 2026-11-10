# Synthetic field-note examples

## Complete synthetic visit
Transcript: Visited Asha Kumari in Demo Nagar village. Age 27 years old. Weight 52 kg. Follow-up required on 2026-11-10.
Expected facts: person_name is Asha Kumari; village is Demo Nagar; age is 27; weight_kg is 52; follow_up_required is true; follow_up_date is 2026-11-10. Evidence should quote the exact transcript wording.

## Missing measurement
Transcript: Today I visited fictional resident Ravi in Sample village. Weight was not recorded.
Expected facts: person_name is Ravi and village is Sample. weight_kg is null. Preserve the unknown measurement; never estimate it.

## Ambiguous weekday
Transcript: A fictional household in Fieldplace needs a follow-up next Monday.
Expected facts: follow_up_required is true, but follow_up_date is null until a calendar date is clarified. Keep the phrase next Monday as evidence and ask one concise question if the date is required.

## Conflicting values
Transcript: Synthetic person Tara is 28 years old; later the note says age 82.
Expected facts: age is uncertain and should be left unresolved for human clarification rather than choosing either number silently.
