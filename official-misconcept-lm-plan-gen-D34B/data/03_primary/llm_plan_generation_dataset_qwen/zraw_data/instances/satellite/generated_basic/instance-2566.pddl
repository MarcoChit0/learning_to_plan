(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	thermograph0 - mode
	thermograph2 - mode
	infrared1 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation3 - direction
	star4 - direction
	groundstation2 - direction
	planet5 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
)
(:goal (and
	(have_image planet5 thermograph2)
))

)
