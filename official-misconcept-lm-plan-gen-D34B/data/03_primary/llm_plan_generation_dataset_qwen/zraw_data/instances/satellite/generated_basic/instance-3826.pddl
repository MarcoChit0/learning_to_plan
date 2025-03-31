(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	image0 - mode
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	star5 - direction
	star3 - direction
	star4 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument1 image0)
	(calibration_target instrument1 star3)
	(supports instrument2 image0)
	(calibration_target instrument2 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
)
(:goal (and
	(pointing satellite0 star5)
	(have_image planet6 image0)
	(have_image planet7 image0)
	(have_image star8 image0)
))

)
