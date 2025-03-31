(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	image0 - mode
	thermograph2 - mode
	thermograph1 - mode
	star0 - direction
	phenomenon1 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon1)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph2)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite1 phenomenon1)
	(have_image phenomenon1 thermograph1)
))

)
