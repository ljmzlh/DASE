WITH sick_audio AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_audio
    WHERE cars.car_id = car_audio.car_id
    AND NLfilter(car_audio.audio_path, 'You are given an audio recording of car diagnostics. Return true if the car from the recording has worn out brakes.')
),
sick_text AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_complaints
    WHERE cars.car_id = car_complaints.car_id
    AND NLfilter(car_complaints.summary, 'In the complaint, the car has some problems with electrical system / connected to electrical system. Complaint: {summary}.')
),
sick_image AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_images
    WHERE cars.car_id = car_images.car_id
    AND NLfilter(car_images.image_path, 'You are given an image of a vehicle or its parts. Return true if car is dented.')
)
SELECT car_id FROM sick_audio
UNION DISTINCT
SELECT car_id FROM sick_text
UNION DISTINCT
SELECT car_id FROM sick_image
