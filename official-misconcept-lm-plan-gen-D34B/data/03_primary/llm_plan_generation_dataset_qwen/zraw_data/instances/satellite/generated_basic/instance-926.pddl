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
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	satellite5 - satellite
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	instrument23 - instrument
	instrument24 - instrument
	spectrograph0 - mode
	infrared1 - mode
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	groundstation2 - direction
	phenomenon4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star3)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 spectrograph0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument11 infrared1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 spectrograph0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation1)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument13 spectrograph0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 spectrograph0)
	(supports instrument14 infrared1)
	(calibration_target instrument14 star3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
	(supports instrument15 infrared1)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 spectrograph0)
	(supports instrument16 infrared1)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 infrared1)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 infrared1)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 groundstation1)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
	(supports instrument19 spectrograph0)
	(supports instrument19 infrared1)
	(calibration_target instrument19 groundstation2)
	(supports instrument20 infrared1)
	(supports instrument20 spectrograph0)
	(calibration_target instrument20 groundstation1)
	(supports instrument21 infrared1)
	(calibration_target instrument21 groundstation2)
	(supports instrument22 spectrograph0)
	(supports instrument22 infrared1)
	(calibration_target instrument22 star3)
	(supports instrument23 infrared1)
	(supports instrument23 spectrograph0)
	(calibration_target instrument23 groundstation2)
	(supports instrument24 infrared1)
	(supports instrument24 spectrograph0)
	(calibration_target instrument24 groundstation2)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(on_board instrument24 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
)
(:goal (and
	(pointing satellite2 groundstation2)
	(pointing satellite5 groundstation2)
	(have_image phenomenon4 infrared1)
	(have_image planet5 infrared1)
))

)
