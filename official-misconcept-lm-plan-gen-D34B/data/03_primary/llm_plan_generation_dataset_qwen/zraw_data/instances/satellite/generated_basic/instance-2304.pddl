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
	satellite2 - satellite
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
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	satellite5 - satellite
	instrument22 - instrument
	instrument23 - instrument
	instrument24 - instrument
	instrument25 - instrument
	instrument26 - instrument
	instrument27 - instrument
	spectrograph0 - mode
	groundstation0 - direction
	groundstation2 - direction
	groundstation1 - direction
	phenomenon3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon3)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 groundstation1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 groundstation1)
	(supports instrument19 spectrograph0)
	(calibration_target instrument19 groundstation0)
	(supports instrument20 spectrograph0)
	(calibration_target instrument20 groundstation0)
	(supports instrument21 spectrograph0)
	(calibration_target instrument21 groundstation0)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon3)
	(supports instrument22 spectrograph0)
	(calibration_target instrument22 groundstation1)
	(supports instrument23 spectrograph0)
	(calibration_target instrument23 groundstation1)
	(supports instrument24 spectrograph0)
	(calibration_target instrument24 groundstation1)
	(supports instrument25 spectrograph0)
	(calibration_target instrument25 groundstation2)
	(supports instrument26 spectrograph0)
	(calibration_target instrument26 groundstation2)
	(supports instrument27 spectrograph0)
	(calibration_target instrument27 groundstation1)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(on_board instrument24 satellite5)
	(on_board instrument25 satellite5)
	(on_board instrument26 satellite5)
	(on_board instrument27 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon3)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite2 phenomenon3)
	(pointing satellite4 phenomenon4)
	(pointing satellite5 groundstation1)
	(have_image phenomenon3 spectrograph0)
	(have_image phenomenon4 spectrograph0)
))

)
