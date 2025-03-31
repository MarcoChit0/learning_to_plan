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
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	spectrograph2 - mode
	image3 - mode
	image1 - mode
	infrared0 - mode
	star0 - direction
	star2 - direction
	star1 - direction
	phenomenon3 - direction
	phenomenon4 - direction
	planet5 - direction
	planet6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument2 spectrograph2)
	(supports instrument2 image3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star2)
	(supports instrument4 image1)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(supports instrument6 image3)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument7 infrared0)
	(supports instrument7 image1)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star1)
	(supports instrument8 image1)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star0)
	(supports instrument9 image3)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star2)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 star1)
	(supports instrument11 image3)
	(supports instrument11 image1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
	(supports instrument12 infrared0)
	(supports instrument12 image1)
	(calibration_target instrument12 star1)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite3 star2)
	(have_image phenomenon3 image3)
	(have_image phenomenon4 image1)
	(have_image planet5 image1)
	(have_image planet6 image1)
	(have_image planet7 infrared0)
))

)
