(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	thermograph3 - mode
	infrared1 - mode
	thermograph0 - mode
	thermograph2 - mode
	groundstation0 - direction
	groundstation4 - direction
	groundstation1 - direction
	groundstation3 - direction
	groundstation2 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument2 thermograph0)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation2)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
)
(:goal (and
	(pointing satellite0 star6)
	(have_image planet5 thermograph0)
	(have_image star6 thermograph2)
	(have_image phenomenon7 thermograph2)
	(have_image phenomenon8 infrared1)
))

)
