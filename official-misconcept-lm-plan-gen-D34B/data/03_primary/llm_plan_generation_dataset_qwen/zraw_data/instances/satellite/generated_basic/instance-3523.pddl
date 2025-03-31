(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
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
	instrument15 - instrument
	instrument16 - instrument
	satellite3 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	spectrograph2 - mode
	spectrograph0 - mode
	spectrograph1 - mode
	star0 - direction
	star1 - direction
	star2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph2)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument5 spectrograph1)
	(supports instrument5 spectrograph0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star1)
	(supports instrument9 spectrograph2)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star0)
	(supports instrument10 spectrograph1)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument11 spectrograph2)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star1)
	(supports instrument12 spectrograph1)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star1)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star1)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 star1)
	(supports instrument15 spectrograph2)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 star0)
	(supports instrument16 spectrograph2)
	(supports instrument16 spectrograph1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star1)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(on_board instrument16 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument17 spectrograph0)
	(supports instrument17 spectrograph1)
	(calibration_target instrument17 star1)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 star0)
	(supports instrument19 spectrograph2)
	(supports instrument19 spectrograph0)
	(calibration_target instrument19 star1)
	(supports instrument20 spectrograph1)
	(supports instrument20 spectrograph2)
	(calibration_target instrument20 star1)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(on_board instrument19 satellite3)
	(on_board instrument20 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(have_image star2 spectrograph2)
	(have_image phenomenon3 spectrograph1)
))

)
