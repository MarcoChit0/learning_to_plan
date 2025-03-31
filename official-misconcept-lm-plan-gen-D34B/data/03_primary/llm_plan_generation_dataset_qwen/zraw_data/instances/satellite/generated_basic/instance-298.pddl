(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	image0 - mode
	thermograph3 - mode
	thermograph1 - mode
	thermograph2 - mode
	star5 - direction
	star1 - direction
	star2 - direction
	star4 - direction
	star3 - direction
	star0 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument1 thermograph3)
	(supports instrument1 image0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 star2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 star4)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
)
(:goal (and
	(have_image star6 thermograph2)
))

)
