(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	thermograph2 - mode
	infrared0 - mode
	thermograph1 - mode
	image3 - mode
	star0 - direction
	star1 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 image3)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph1)
	(supports instrument3 image3)
	(calibration_target instrument3 star0)
	(supports instrument4 image3)
	(supports instrument4 thermograph2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image star1 image3)
))

)
