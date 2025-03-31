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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite4 - satellite
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	image1 - mode
	spectrograph3 - mode
	spectrograph0 - mode
	image2 - mode
	star1 - direction
	star0 - direction
	planet2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph3)
	(supports instrument0 image2)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument2 image2)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star0)
	(supports instrument4 image2)
	(supports instrument4 spectrograph3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 star0)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(supports instrument7 image2)
	(supports instrument7 image1)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
	(supports instrument8 spectrograph3)
	(supports instrument8 image2)
	(calibration_target instrument8 star0)
	(supports instrument9 image2)
	(calibration_target instrument9 star0)
	(supports instrument10 image2)
	(supports instrument10 spectrograph3)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star0)
	(supports instrument11 spectrograph0)
	(supports instrument11 spectrograph3)
	(calibration_target instrument11 star1)
	(supports instrument12 image1)
	(supports instrument12 spectrograph0)
	(supports instrument12 spectrograph3)
	(calibration_target instrument12 star1)
	(supports instrument13 image1)
	(supports instrument13 image2)
	(supports instrument13 spectrograph3)
	(calibration_target instrument13 star0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star1)
	(supports instrument15 image1)
	(supports instrument15 spectrograph3)
	(supports instrument15 image2)
	(calibration_target instrument15 star0)
	(supports instrument16 spectrograph3)
	(calibration_target instrument16 star0)
	(supports instrument17 image1)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 star0)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet2)
	(supports instrument18 spectrograph3)
	(calibration_target instrument18 star1)
	(supports instrument19 image2)
	(calibration_target instrument19 star1)
	(supports instrument20 spectrograph3)
	(supports instrument20 image1)
	(calibration_target instrument20 star0)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite1 star0)
	(pointing satellite3 star0)
	(have_image planet2 spectrograph3)
	(have_image phenomenon3 spectrograph0)
))

)
