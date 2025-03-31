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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	spectrograph3 - mode
	image2 - mode
	infrared1 - mode
	image0 - mode
	star0 - direction
	phenomenon1 - direction
	planet2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument2 image2)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph3)
	(supports instrument3 infrared1)
	(supports instrument3 image2)
	(calibration_target instrument3 star0)
	(supports instrument4 image2)
	(supports instrument4 spectrograph3)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument5 image0)
	(supports instrument5 image2)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
	(supports instrument7 spectrograph3)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared1)
	(supports instrument8 image2)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph3)
	(supports instrument9 image0)
	(supports instrument9 image2)
	(calibration_target instrument9 star0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(have_image phenomenon1 image2)
	(have_image planet2 spectrograph3)
	(have_image phenomenon3 infrared1)
))

)
