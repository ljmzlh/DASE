SELECT 2026 - AVG(cars.year) AS average_age
FROM cars, car_complaints
WHERE cars.car_id = car_complaints.car_id
AND NLfilter(car_complaints.summary, 'In the complaint, the car has some problems with engine / connected to engine. Complaint: {summary}.')
