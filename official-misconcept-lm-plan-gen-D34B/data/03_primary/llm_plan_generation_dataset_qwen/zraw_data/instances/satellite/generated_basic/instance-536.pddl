(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	image0 - mode
	image2 - mode
	infrared1 - mode
	thermograph3 - mode
	star0 - direction
	groundstation3 - direction
	groundstation4 - direction
	star2 - direction
	groundstation1 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph3)
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 image2)
	(supports instrument1 thermograph3)
	(supports instrument1 image0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(have_image planet5 image0)
	(have_image phenomenon6 infrared1)
	(have_image phenomenon7 infrared1)
	(have_image planet8 thermograph3)
))

)
