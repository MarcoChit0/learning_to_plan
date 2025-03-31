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
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	thermograph2 - mode
	image3 - mode
	image0 - mode
	spectrograph1 - mode
	star0 - direction
	phenomenon1 - direction
	planet2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 image0)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 image3)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph2)
	(supports instrument6 image3)
	(calibration_target instrument6 star0)
	(supports instrument7 image3)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument8 image0)
	(supports instrument8 thermograph2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star0)
	(supports instrument9 image3)
	(calibration_target instrument9 star0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument10 image3)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star0)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon1)
)
(:goal (and
	(pointing satellite1 planet2)
	(pointing satellite2 planet2)
	(have_image phenomenon1 spectrograph1)
	(have_image planet2 thermograph2)
	(have_image phenomenon3 spectrograph1)
))

)
