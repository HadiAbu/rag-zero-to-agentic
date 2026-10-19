---
title: Payment Gateway Error Catalog
doc_id: error-catalog
department: Engineering
access: internal
version: 5
status: current
effective_date: 2026-07-01
---
# Payment Gateway Error Catalog

## ERR-5001

Gateway timeout while authorizing a card payment. Safe to retry with backoff.

## ERR-5002

Gateway timeout while capturing an authorized payment. Retry once, then check the capture status before charging again.

## ERR-5004

Gateway timeout while refunding a payment. Do not retry; the refund may already be queued. Open a finance ticket.

## ERR-5006

Gateway timeout while voiding an authorization. Retry up to 2 times.

## ERR-5007

Gateway timeout while creating a payout to a merchant. Never retry automatically; payouts must be checked by the treasury team first.

## ERR-5009

Gateway timeout while fetching a payment status. Safe to retry immediately.
