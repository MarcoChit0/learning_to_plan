(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	infrared3 - mode
	image2 - mode
	spectrograph1 - mode
	infrared0 - mode
	star0 - direction
	phenomenon1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 infrared0)
	(supports instrument0 infrared3)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon1)
	(supports instrument1 infrared3)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared3)
	(supports instrument2 infrared0)
	(supports instrument2 image2)
	(calibration_target instrument2 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument3 spectrograph1)
	(supports instrument3 infrared3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared0)
	(supports instrument5 spectrograph1)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared0)
	(supports instrument6 spectrograph1)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon3)
	(supports instrument8 infrared0)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star0)
	(supports instrument10 image2)
	(supports instrument10 infrared3)
	(calibration_target instrument10 star0)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon3)
)
(:goal (and
	(pointing satellite1 phenomenon2)
	(pointing satellite3 phenomenon2)
	(have_image phenomenon1 infrared0)
	(have_image phenomenon2 image2)
	(have_image phenomenon3 image2)
))

)
