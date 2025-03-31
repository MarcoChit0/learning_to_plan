(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	thermograph2 - mode
	infrared0 - mode
	spectrograph1 - mode
	groundstation1 - direction
	star3 - direction
	star4 - direction
	star0 - direction
	groundstation2 - direction
	planet5 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(have_image planet5 infrared0)
))

)
