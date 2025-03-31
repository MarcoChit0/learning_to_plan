(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	spectrograph2 - mode
	spectrograph3 - mode
	image0 - mode
	thermograph1 - mode
	star0 - direction
	phenomenon1 - direction
	star2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(supports instrument1 spectrograph3)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph2)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon1)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph3)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph3)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite2 star2)
	(have_image phenomenon1 spectrograph3)
	(have_image star2 thermograph1)
	(have_image phenomenon3 thermograph1)
))

)
