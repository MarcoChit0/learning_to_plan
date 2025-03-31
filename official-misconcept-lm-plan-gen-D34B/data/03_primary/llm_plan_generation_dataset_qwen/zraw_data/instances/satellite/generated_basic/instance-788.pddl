(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	image1 - mode
	image0 - mode
	groundstation1 - direction
	star3 - direction
	star0 - direction
	star2 - direction
	groundstation4 - direction
	star5 - direction
	star6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(have_image star5 image1)
	(have_image star6 image0)
	(have_image planet7 image0)
	(have_image star8 image0)
))

)
