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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	satellite5 - satellite
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	spectrograph0 - mode
	groundstation1 - direction
	star3 - direction
	star4 - direction
	star0 - direction
	groundstation2 - direction
	groundstation5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 star4)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star4)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 star4)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star3)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star4)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation1)
	(calibration_target instrument13 star0)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star0)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star3)
	(calibration_target instrument15 groundstation2)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star0)
	(calibration_target instrument16 groundstation5)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 groundstation5)
	(calibration_target instrument17 star0)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 groundstation5)
	(supports instrument19 spectrograph0)
	(calibration_target instrument19 star4)
	(supports instrument20 spectrograph0)
	(calibration_target instrument20 star4)
	(supports instrument21 spectrograph0)
	(calibration_target instrument21 groundstation2)
	(calibration_target instrument21 star0)
	(supports instrument22 spectrograph0)
	(calibration_target instrument22 groundstation5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon6)
)
(:goal (and
	(have_image phenomenon6 spectrograph0)
))

)
