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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	spectrograph0 - mode
	star3 - direction
	groundstation1 - direction
	groundstation2 - direction
	groundstation0 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star3)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star3)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon4)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 groundstation2)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 groundstation2)
	(supports instrument19 spectrograph0)
	(calibration_target instrument19 groundstation0)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star3)
)
(:goal (and
	(pointing satellite3 phenomenon4)
	(pointing satellite5 star3)
	(have_image phenomenon4 spectrograph0)
))

)
