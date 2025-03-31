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
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite5 - satellite
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	spectrograph0 - mode
	star0 - direction
	groundstation1 - direction
	star4 - direction
	groundstation2 - direction
	groundstation3 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star4)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star4)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star4)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet7)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star4)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 groundstation3)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 groundstation3)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon6)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 groundstation3)
	(supports instrument19 spectrograph0)
	(calibration_target instrument19 groundstation2)
	(supports instrument20 spectrograph0)
	(calibration_target instrument20 groundstation3)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon6)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(pointing satellite2 groundstation1)
	(pointing satellite3 star5)
	(pointing satellite4 star0)
	(have_image star5 spectrograph0)
	(have_image phenomenon6 spectrograph0)
	(have_image planet7 spectrograph0)
))

)
