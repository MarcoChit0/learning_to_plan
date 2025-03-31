(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	thermograph2 - mode
	thermograph0 - mode
	image1 - mode
	star5 - direction
	star0 - direction
	groundstation1 - direction
	star4 - direction
	star2 - direction
	groundstation3 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation1)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 star2)
	(supports instrument3 image1)
	(calibration_target instrument3 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 star2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
)
(:goal (and
	(pointing satellite1 star7)
	(have_image phenomenon6 thermograph0)
	(have_image star7 thermograph2)
	(have_image phenomenon8 image1)
))

)
