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
	satellite2 - satellite
	instrument5 - instrument
	thermograph1 - mode
	infrared0 - mode
	image2 - mode
	groundstation1 - direction
	star0 - direction
	planet2 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star0)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet2)
	(supports instrument3 infrared0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument5 thermograph1)
	(supports instrument5 infrared0)
	(supports instrument5 image2)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(have_image planet2 thermograph1)
	(have_image star3 infrared0)
	(have_image planet4 thermograph1)
))

)
