(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	infrared0 - mode
	image2 - mode
	image1 - mode
	star0 - direction
	star4 - direction
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	star5 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image2)
	(supports instrument0 image1)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 image2)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
)
(:goal (and
	(pointing satellite0 star8)
	(pointing satellite1 star5)
	(have_image star5 infrared0)
	(have_image planet6 infrared0)
	(have_image phenomenon7 image1)
	(have_image star8 infrared0)
	(have_image star9 image1)
))

)
