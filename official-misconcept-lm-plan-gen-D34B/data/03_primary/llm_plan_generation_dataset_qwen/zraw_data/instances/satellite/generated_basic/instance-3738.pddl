(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite4 - satellite
	instrument17 - instrument
	instrument18 - instrument
	satellite5 - satellite
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	infrared0 - mode
	spectrograph2 - mode
	image1 - mode
	star1 - direction
	star2 - direction
	star0 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star2)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star2)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument6 infrared0)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph2)
	(supports instrument7 infrared0)
	(supports instrument7 image1)
	(calibration_target instrument7 star2)
	(supports instrument8 image1)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star2)
	(supports instrument9 infrared0)
	(supports instrument9 image1)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star2)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument10 spectrograph2)
	(supports instrument10 image1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star1)
	(supports instrument11 image1)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 star1)
	(supports instrument12 image1)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star1)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 star0)
	(supports instrument14 image1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star2)
	(supports instrument15 image1)
	(supports instrument15 spectrograph2)
	(calibration_target instrument15 star0)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star0)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument17 infrared0)
	(supports instrument17 spectrograph2)
	(supports instrument17 image1)
	(calibration_target instrument17 star2)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star2)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon3)
	(supports instrument19 infrared0)
	(supports instrument19 spectrograph2)
	(calibration_target instrument19 star0)
	(supports instrument20 spectrograph2)
	(calibration_target instrument20 star2)
	(supports instrument21 infrared0)
	(calibration_target instrument21 star0)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star1)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 phenomenon3)
	(pointing satellite5 star1)
	(have_image phenomenon3 infrared0)
))

)
