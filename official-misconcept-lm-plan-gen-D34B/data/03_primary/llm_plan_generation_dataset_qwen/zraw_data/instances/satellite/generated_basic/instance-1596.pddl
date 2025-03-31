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
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	satellite5 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	spectrograph0 - mode
	star1 - direction
	star3 - direction
	groundstation0 - direction
	star4 - direction
	groundstation2 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star4)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star4)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star4)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star4)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star4)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation2)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star4)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star4)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star4)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 groundstation2)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star4)
)
(:goal (and
	(pointing satellite2 groundstation0)
	(pointing satellite3 planet7)
	(pointing satellite4 phenomenon6)
	(pointing satellite5 groundstation2)
	(have_image phenomenon5 spectrograph0)
	(have_image phenomenon6 spectrograph0)
	(have_image planet7 spectrograph0)
))

)
