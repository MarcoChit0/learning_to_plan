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
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite4 - satellite
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	spectrograph0 - mode
	image1 - mode
	star0 - direction
	phenomenon1 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument6 spectrograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(supports instrument7 image1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 star0)
	(supports instrument9 image1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star0)
	(supports instrument10 spectrograph0)
	(supports instrument10 image1)
	(calibration_target instrument10 star0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument11 spectrograph0)
	(supports instrument11 image1)
	(calibration_target instrument11 star0)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star0)
	(supports instrument13 spectrograph0)
	(supports instrument13 image1)
	(calibration_target instrument13 star0)
	(supports instrument14 image1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star0)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon1)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star0)
	(supports instrument16 image1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star0)
	(supports instrument17 image1)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 star0)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon1)
	(supports instrument18 image1)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 star0)
	(supports instrument19 spectrograph0)
	(supports instrument19 image1)
	(calibration_target instrument19 star0)
	(supports instrument20 image1)
	(supports instrument20 spectrograph0)
	(calibration_target instrument20 star0)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite2 star0)
	(have_image phenomenon1 image1)
))

)
