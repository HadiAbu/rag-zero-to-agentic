# Candidate CVs (the challenge data)

All people, companies, emails and phone numbers here are **invented** for the lab.

| Folder | What | Access |
|--------|------|--------|
| `candidates/` | One CV per candidate (`cand-004.md`). Two candidates also have an **outdated** CV (`status: superseded`) | internal |
| `hr-private/` | Contact details, salary expectation, notice period, recruiter note | **restricted**, `allowed_roles: hr` |
| `jobs/` | Three open positions with must-have and nice-to-have requirements | internal |

Every CV header carries `candidate_id`, `location`, `years`, `headline`, `languages`, `status`
and `updated`, so you can filter on them (for example: Haifa, at least 5 years).

To add your own CVs, copy one file and keep the same header. Never put real people's CVs in a public repo.
