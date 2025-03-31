(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	spectrograph2 - mode
	thermograph0 - mode
	spectrograph3 - mode
	thermograph1 - mode
	star2 - direction
	star0 - direction
	groundstation1 - direction
	groundstation3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 thermograph1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(have_image phenomenon4 spectrograph2)
))

)
