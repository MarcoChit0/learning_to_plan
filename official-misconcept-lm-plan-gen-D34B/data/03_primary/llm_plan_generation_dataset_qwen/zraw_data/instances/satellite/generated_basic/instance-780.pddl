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
	star0 - direction
	groundstation2 - direction
	star3 - direction
	star4 - direction
	star1 - direction
	phenomenon5 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star4)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
)
(:goal (and
	(have_image phenomenon5 image1)
	(have_image star6 thermograph0)
	(have_image planet7 thermograph0)
	(have_image planet8 thermograph0)
))

)
