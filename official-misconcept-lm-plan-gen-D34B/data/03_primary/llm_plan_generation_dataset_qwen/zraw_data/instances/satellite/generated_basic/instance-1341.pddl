(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	infrared0 - mode
	thermograph1 - mode
	image2 - mode
	thermograph3 - mode
	star0 - direction
	planet1 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph3)
	(supports instrument2 image2)
	(calibration_target instrument2 star0)
	(supports instrument3 image2)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(have_image planet1 infrared0)
))

)
