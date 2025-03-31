(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	thermograph0 - mode
	spectrograph2 - mode
	infrared1 - mode
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	groundstation4 - direction
	groundstation3 - direction
	planet5 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument1 spectrograph2)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(pointing satellite0 planet5)
	(have_image planet5 thermograph0)
))

)
