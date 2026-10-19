---
title: Payments Service Runbook
doc_id: payments-runbook
department: Engineering
access: internal
version: 12
status: current
effective_date: 2026-08-20
---
# Payments Service Runbook

## 1. Error ERR-4012

ERR-4012 means the card was declined by the issuing bank. Do not retry automatically. Show the customer a message asking them to use another card or contact their bank.

## 2. Error ERR-5003

ERR-5003 means the payment gateway timed out. Retry up to 3 times with exponential backoff (1 s, 2 s, 4 s). If all retries fail, mark the order as pending and alert the on-call engineer.

## 3. Error ERR-5100

ERR-5100 means a duplicate transaction was detected. Never charge again; look up the original transaction ID and return its status.

## 4. Escalation

Escalate to the payments on-call engineer through the paging tool if more than 5% of payments fail within 10 minutes.
