SELECT car_id
FROM car_images
WHERE NLfilter(image_path, 'You are given an image of a vehicle or its parts. Return true if car has both, puncture and paint scratches.')
LIMIT 100
