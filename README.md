# VIP Scoring API — Priority Grounded (FINAL)
Priority: session CSV -> global CSV -> base CSV.

POST /score:
{
  "session_id": "s1",
  "rows": [
    {"lead_id":"L-1","order_count":3,"revenue":1200,"currency":"USD","notes":"..."},
    {"lead_id":"L-2","order_count":1,"revenue":200}
  ]
}

POST /chat:
{
  "session_id":"s1",
  "question":"Tổng kết giúp mình",
  "rows":[{"lead_id":"L-10","order_count":2,"revenue":500}]
}
