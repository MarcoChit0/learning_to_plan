(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	thermograph1 - mode
	infrared0 - mode
	image2 - mode
	groundstation1 - direction
	star0 - direction
	planet2 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 thermograph1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
	(supports instrument3 image2)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 image2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image planet2 thermograph1)
))

)
