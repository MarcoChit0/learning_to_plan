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
	satellite2 - satellite
	instrument8 - instrument
	spectrograph3 - mode
	spectrograph2 - mode
	thermograph1 - mode
	thermograph0 - mode
	groundstation3 - direction
	star0 - direction
	star1 - direction
	star2 - direction
	planet4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument5 spectrograph2)
	(supports instrument5 thermograph0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph0)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 star1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument8 thermograph0)
	(supports instrument8 thermograph1)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 star2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image planet4 spectrograph2)
	(have_image planet5 spectrograph2)
	(have_image phenomenon6 thermograph1)
))

)
