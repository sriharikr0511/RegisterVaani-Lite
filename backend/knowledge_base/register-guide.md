# RegisterVaani synthetic register guide

## Purpose and data safety
This illustrative register supports data entry, not diagnosis or clinical decisions. Use only synthetic information in demonstrations. Field workers should review every AI suggestion and approve records only after checking the source note.

## Register field definitions
A visit record may include visit_date, worker_name, household_id, person_name, age, village, beneficiary_type, weight_kg, follow_up_required, follow_up_date, and notes. Unknown values must be left blank (null). The illustrative register requires person_name and village before approval. A missing weight is allowed and must not be estimated.

## Supported beneficiary types
Examples of beneficiary type labels include ASHA, Anganwadi worker, pregnant woman, child, and senior citizen. Keep the wording stated by the source note. Do not infer a beneficiary type from age, name, or measurements.

## Human review and approval
The transcript is the evidence source. Keep a short exact quote for each populated value. A user correction should be labeled as clarification or human input, not presented as a transcript quote. Drafts are not approved records. Only the explicit Approve action finalizes a record.
