(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	spectrograph1 - mode
	spectrograph2 - mode
	thermograph0 - mode
	star0 - direction
	planet1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph2)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph1)
	(supports instrument6 spectrograph2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph0)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
)
(:goal (and
	(have_image planet1 thermograph0)
	(have_image phenomenon2 spectrograph2)
))

)
