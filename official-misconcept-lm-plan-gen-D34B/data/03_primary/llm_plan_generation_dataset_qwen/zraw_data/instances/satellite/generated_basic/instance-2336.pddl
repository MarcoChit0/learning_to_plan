(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	thermograph1 - mode
	thermograph0 - mode
	spectrograph2 - mode
	star0 - direction
	groundstation1 - direction
	star3 - direction
	star2 - direction
	star4 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(pointing satellite0 star2)
	(have_image star5 spectrograph2)
	(have_image star6 thermograph1)
	(have_image phenomenon7 thermograph0)
	(have_image phenomenon8 thermograph1)
	(have_image star9 spectrograph2)
))

)
