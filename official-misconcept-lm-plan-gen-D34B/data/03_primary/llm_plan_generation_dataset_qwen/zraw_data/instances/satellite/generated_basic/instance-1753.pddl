(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
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
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	spectrograph1 - mode
	spectrograph0 - mode
	star2 - direction
	star1 - direction
	star0 - direction
	star3 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 spectrograph1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument10 spectrograph1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star2)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star2)
	(supports instrument12 spectrograph0)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 star2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 star2)
	(supports instrument14 spectrograph1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star0)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument15 spectrograph1)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star2)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star1)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument17 spectrograph1)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 star2)
	(supports instrument18 spectrograph0)
	(supports instrument18 spectrograph1)
	(calibration_target instrument18 star1)
	(supports instrument19 spectrograph1)
	(calibration_target instrument19 star1)
	(supports instrument20 spectrograph0)
	(calibration_target instrument20 star0)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star2)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite5 star0)
	(have_image star3 spectrograph0)
))

)
