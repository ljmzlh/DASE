SELECT cars.car_id
FROM cars, car_images
WHERE cars.car_id = car_images.car_id
AND cars.transmission = 'Manual'
AND NLfilter(car_images.image_path, 'You are given an image of a vehicle or its parts. Return true if car is not damaged.')
LIMIT 10
