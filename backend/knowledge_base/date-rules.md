# Synthetic date and follow-up rules

## Date format
Store an explicitly supplied visit date or follow-up date as ISO YYYY-MM-DD. Accept unambiguous numeric day/month/year or written month dates only when the day, month, and year can be determined. If a note says only a weekday such as next Monday, do not invent a calendar date; keep the phrase as evidence and ask the worker to clarify the exact date when required.

## Follow-up fields
follow_up_required may be true only when the worker explicitly says that follow-up, revisit, or return is needed. It may be false only if the note explicitly says no follow-up is needed. Otherwise use null. follow_up_date must remain null when no exact date is available.

## Illustrative data-quality ranges
For this demo only, age must be numeric from 0 through 120, and weight must be numeric from 0.1 through 300 kg. These configurable illustrative bounds are not official standards or medical advice. Invalid values should be flagged for human review rather than silently changed.
