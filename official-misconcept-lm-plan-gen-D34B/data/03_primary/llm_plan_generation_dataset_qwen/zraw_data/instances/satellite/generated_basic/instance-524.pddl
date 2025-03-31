(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	infrared2 - mode
	thermograph3 - mode
	infrared0 - mode
	image1 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	groundstation4 - direction
	star3 - direction
	planet5 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument2 thermograph3)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star3)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
)
(:goal (and
	(have_image planet5 image1)
	(have_image planet6 thermograph3)
	(have_image star7 image1)
	(have_image planet8 thermograph3)
))

)
