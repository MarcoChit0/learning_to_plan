(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	spectrograph0 - mode
	spectrograph2 - mode
	image1 - mode
	thermograph3 - mode
	star0 - direction
	phenomenon1 - direction
	planet2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet2)
	(supports instrument3 image1)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 spectrograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph3)
	(supports instrument8 image1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
)
(:goal (and
	(pointing satellite2 star0)
	(have_image phenomenon1 spectrograph0)
	(have_image planet2 image1)
	(have_image planet3 thermograph3)
))

)
