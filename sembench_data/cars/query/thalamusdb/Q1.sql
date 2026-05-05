SELECT DISTINCT car_id
FROM car_complaints
WHERE NLfilter(summary, 'You are be given a textual complaint entailing that the car was in a crash/accident/collision. Complaint: {summary}.')
