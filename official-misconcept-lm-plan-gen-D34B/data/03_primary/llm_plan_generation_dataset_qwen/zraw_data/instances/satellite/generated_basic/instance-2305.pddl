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
	satellite2 - satellite
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
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	instrument23 - instrument
	satellite5 - satellite
	instrument24 - instrument
	instrument25 - instrument
	instrument26 - instrument
	spectrograph0 - mode
	groundstation1 - direction
	star2 - direction
	groundstation0 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star2)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation1)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 groundstation0)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 spectrograph0)
	(calibration_target instrument19 groundstation1)
	(supports instrument20 spectrograph0)
	(calibration_target instrument20 groundstation1)
	(supports instrument21 spectrograph0)
	(calibration_target instrument21 star2)
	(supports instrument22 spectrograph0)
	(calibration_target instrument22 groundstation0)
	(supports instrument23 spectrograph0)
	(calibration_target instrument23 star2)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(on_board instrument22 satellite4)
	(on_board instrument23 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
	(supports instrument24 spectrograph0)
	(calibration_target instrument24 star2)
	(supports instrument25 spectrograph0)
	(calibration_target instrument25 star2)
	(supports instrument26 spectrograph0)
	(calibration_target instrument26 groundstation0)
	(on_board instrument24 satellite5)
	(on_board instrument25 satellite5)
	(on_board instrument26 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star2)
)
(:goal (and
	(have_image star3 spectrograph0)
	(have_image planet4 spectrograph0)
))

)
