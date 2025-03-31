(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	satellite5 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	image0 - mode
	spectrograph2 - mode
	spectrograph1 - mode
	groundstation2 - direction
	star1 - direction
	groundstation0 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 spectrograph1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star1)
	(supports instrument4 image0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 spectrograph2)
	(supports instrument5 spectrograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 spectrograph2)
	(supports instrument6 image0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image0)
	(supports instrument9 spectrograph2)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star1)
	(supports instrument10 image0)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image0)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 star1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 spectrograph2)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument14 spectrograph1)
	(supports instrument14 spectrograph2)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation0)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
	(supports instrument15 spectrograph2)
	(calibration_target instrument15 star1)
	(supports instrument16 spectrograph2)
	(supports instrument16 spectrograph1)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 image0)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 image0)
	(supports instrument18 spectrograph2)
	(supports instrument18 spectrograph1)
	(calibration_target instrument18 star1)
	(supports instrument19 spectrograph1)
	(supports instrument19 image0)
	(calibration_target instrument19 star1)
	(supports instrument20 spectrograph1)
	(supports instrument20 image0)
	(calibration_target instrument20 groundstation0)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation2)
)
(:goal (and
	(pointing satellite1 star1)
	(have_image planet3 spectrograph1)
))

)
