(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	thermograph2 - mode
	infrared1 - mode
	infrared0 - mode
	star0 - direction
	star1 - direction
	star2 - direction
	star3 - direction
	groundstation4 - direction
	phenomenon5 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation4)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite1 phenomenon8)
	(have_image phenomenon5 infrared0)
	(have_image planet6 thermograph2)
	(have_image star7 thermograph2)
	(have_image phenomenon8 thermograph2)
	(have_image planet9 thermograph2)
))

)
