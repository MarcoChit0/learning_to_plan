(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	thermograph1 - mode
	image2 - mode
	infrared0 - mode
	star0 - direction
	groundstation2 - direction
	groundstation3 - direction
	star4 - direction
	groundstation1 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 thermograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image2)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star4)
	(supports instrument3 image2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
)
(:goal (and
	(pointing satellite0 planet8)
	(have_image star5 thermograph1)
	(have_image star6 image2)
	(have_image star7 image2)
	(have_image planet8 image2)
	(have_image phenomenon9 image2)
))

)
