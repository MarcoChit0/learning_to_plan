(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	thermograph2 - mode
	thermograph0 - mode
	image1 - mode
	groundstation2 - direction
	groundstation3 - direction
	groundstation1 - direction
	star0 - direction
	groundstation4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image1)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
)
(:goal (and
	(pointing satellite0 phenomenon5)
	(have_image phenomenon5 image1)
	(have_image phenomenon6 image1)
	(have_image star7 image1)
	(have_image planet8 image1)
	(have_image planet9 thermograph0)
))

)
