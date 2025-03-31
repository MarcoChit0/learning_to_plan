(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	thermograph0 - mode
	image1 - mode
	star1 - direction
	star3 - direction
	groundstation4 - direction
	star2 - direction
	star0 - direction
	phenomenon5 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(have_image phenomenon5 image1)
	(have_image star6 image1)
	(have_image planet7 image1)
	(have_image phenomenon8 thermograph0)
))

)
