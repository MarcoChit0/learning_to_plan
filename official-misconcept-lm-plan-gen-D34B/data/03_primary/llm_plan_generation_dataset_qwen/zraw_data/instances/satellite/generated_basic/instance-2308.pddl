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
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	spectrograph0 - mode
	star2 - direction
	star1 - direction
	groundstation0 - direction
	phenomenon3 - direction
	star4 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star2)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon3)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star1)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star2)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 star1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star2)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star1)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 star1)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 star1)
	(supports instrument19 spectrograph0)
	(calibration_target instrument19 star1)
	(supports instrument20 spectrograph0)
	(calibration_target instrument20 groundstation0)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star2)
)
(:goal (and
	(pointing satellite4 star1)
	(have_image phenomenon3 spectrograph0)
	(have_image star4 spectrograph0)
))

)
