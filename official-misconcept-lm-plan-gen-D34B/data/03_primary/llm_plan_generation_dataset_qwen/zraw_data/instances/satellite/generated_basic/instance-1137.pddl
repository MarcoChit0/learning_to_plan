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
	satellite5 - satellite
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	spectrograph0 - mode
	star4 - direction
	star5 - direction
	groundstation1 - direction
	star2 - direction
	star0 - direction
	groundstation3 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 star4)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 star2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation3)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star4)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star5)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star5)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 star0)
	(calibration_target instrument13 star4)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation1)
	(calibration_target instrument14 star0)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star5)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 groundstation3)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 star0)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 star0)
	(supports instrument19 spectrograph0)
	(calibration_target instrument19 star2)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
	(supports instrument20 spectrograph0)
	(calibration_target instrument20 star2)
	(calibration_target instrument20 groundstation1)
	(supports instrument21 spectrograph0)
	(calibration_target instrument21 star0)
	(calibration_target instrument21 star2)
	(supports instrument22 spectrograph0)
	(calibration_target instrument22 groundstation3)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(have_image phenomenon6 spectrograph0)
))

)
