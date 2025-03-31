(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite4 - satellite
	instrument17 - instrument
	image1 - mode
	spectrograph3 - mode
	image2 - mode
	image0 - mode
	groundstation1 - direction
	star4 - direction
	star0 - direction
	groundstation2 - direction
	star3 - direction
	star5 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
	star10 - direction
	star11 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument1 spectrograph3)
	(supports instrument1 image1)
	(calibration_target instrument1 star4)
	(supports instrument2 spectrograph3)
	(supports instrument2 image0)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 star5)
	(supports instrument4 image2)
	(calibration_target instrument4 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 star3)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image2)
	(calibration_target instrument7 star4)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 star5)
	(supports instrument9 image0)
	(calibration_target instrument9 star4)
	(supports instrument10 image2)
	(supports instrument10 image0)
	(calibration_target instrument10 star5)
	(calibration_target instrument10 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument11 image1)
	(supports instrument11 spectrograph3)
	(calibration_target instrument11 star5)
	(supports instrument12 image0)
	(supports instrument12 spectrograph3)
	(calibration_target instrument12 star3)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 image2)
	(supports instrument13 spectrograph3)
	(calibration_target instrument13 star0)
	(calibration_target instrument13 star4)
	(supports instrument14 image0)
	(supports instrument14 image1)
	(supports instrument14 image2)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 image0)
	(calibration_target instrument15 star3)
	(supports instrument16 spectrograph3)
	(supports instrument16 image1)
	(supports instrument16 image2)
	(calibration_target instrument16 star5)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
	(supports instrument17 spectrograph3)
	(calibration_target instrument17 star5)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star11)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite2 groundstation2)
	(pointing satellite4 star9)
	(have_image phenomenon6 image2)
	(have_image star7 image2)
	(have_image phenomenon8 image0)
	(have_image star9 image2)
	(have_image star10 image2)
	(have_image star11 image1)
))

)
