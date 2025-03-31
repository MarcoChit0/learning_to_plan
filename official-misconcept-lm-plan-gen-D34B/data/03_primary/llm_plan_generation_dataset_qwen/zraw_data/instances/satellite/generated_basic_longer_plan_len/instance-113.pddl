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
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	satellite5 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	infrared0 - mode
	image1 - mode
	spectrograph2 - mode
	star3 - direction
	star0 - direction
	star2 - direction
	star1 - direction
	star4 - direction
	star5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
	star9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star2)
	(supports instrument1 image1)
	(calibration_target instrument1 star1)
	(supports instrument2 image1)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star3)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star3)
	(supports instrument8 image1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star3)
	(supports instrument9 infrared0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument10 image1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star3)
	(supports instrument11 image1)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 star0)
	(supports instrument12 spectrograph2)
	(supports instrument12 image1)
	(calibration_target instrument12 star1)
	(supports instrument13 spectrograph2)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star8)
	(supports instrument14 infrared0)
	(supports instrument14 spectrograph2)
	(supports instrument14 image1)
	(calibration_target instrument14 star3)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
	(supports instrument15 infrared0)
	(supports instrument15 image1)
	(supports instrument15 spectrograph2)
	(calibration_target instrument15 star0)
	(supports instrument16 infrared0)
	(supports instrument16 image1)
	(calibration_target instrument16 star0)
	(supports instrument17 spectrograph2)
	(supports instrument17 infrared0)
	(supports instrument17 image1)
	(calibration_target instrument17 star2)
	(supports instrument18 image1)
	(calibration_target instrument18 star2)
	(supports instrument19 image1)
	(supports instrument19 spectrograph2)
	(calibration_target instrument19 star1)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star4)
)
(:goal (and
	(pointing satellite3 star0)
	(pointing satellite4 phenomenon7)
	(have_image star4 spectrograph2)
	(have_image star5 infrared0)
	(have_image phenomenon6 spectrograph2)
	(have_image phenomenon7 infrared0)
	(have_image star8 spectrograph2)
	(have_image star9 spectrograph2)
	(have_image phenomenon10 image1)
))

)
