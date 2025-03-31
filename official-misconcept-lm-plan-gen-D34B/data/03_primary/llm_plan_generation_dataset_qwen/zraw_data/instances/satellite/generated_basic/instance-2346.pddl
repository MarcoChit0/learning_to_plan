(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	spectrograph2 - mode
	spectrograph0 - mode
	thermograph1 - mode
	star1 - direction
	star3 - direction
	star2 - direction
	star0 - direction
	star4 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 thermograph1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(have_image star4 thermograph1)
	(have_image phenomenon5 spectrograph0)
	(have_image star6 spectrograph2)
	(have_image star7 spectrograph2)
	(have_image phenomenon8 spectrograph2)
	(have_image star9 thermograph1)
))

)
