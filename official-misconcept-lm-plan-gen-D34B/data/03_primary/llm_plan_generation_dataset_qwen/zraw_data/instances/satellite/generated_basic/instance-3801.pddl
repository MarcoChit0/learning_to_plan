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
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	satellite4 - satellite
	instrument19 - instrument
	satellite5 - satellite
	instrument20 - instrument
	image0 - mode
	thermograph1 - mode
	spectrograph2 - mode
	spectrograph3 - mode
	groundstation1 - direction
	star2 - direction
	groundstation0 - direction
	star3 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star3)
	(supports instrument1 image0)
	(supports instrument1 spectrograph3)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph3)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph3)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument6 image0)
	(supports instrument6 spectrograph2)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star3)
	(supports instrument7 spectrograph2)
	(supports instrument7 spectrograph3)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph2)
	(supports instrument8 thermograph1)
	(supports instrument8 image0)
	(calibration_target instrument8 star3)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star3)
	(supports instrument10 spectrograph3)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 star3)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument11 thermograph1)
	(supports instrument11 spectrograph3)
	(calibration_target instrument11 star3)
	(supports instrument12 spectrograph3)
	(calibration_target instrument12 star2)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image0)
	(calibration_target instrument14 star3)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument15 spectrograph3)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star3)
	(supports instrument16 spectrograph3)
	(supports instrument16 thermograph1)
	(calibration_target instrument16 star2)
	(supports instrument17 image0)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 star3)
	(supports instrument18 thermograph1)
	(supports instrument18 spectrograph3)
	(supports instrument18 spectrograph2)
	(calibration_target instrument18 groundstation0)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument19 spectrograph2)
	(supports instrument19 thermograph1)
	(calibration_target instrument19 star3)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
	(supports instrument20 thermograph1)
	(calibration_target instrument20 star3)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star2)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite2 planet5)
	(pointing satellite3 groundstation0)
	(pointing satellite4 planet5)
	(have_image star4 spectrograph2)
	(have_image planet5 spectrograph3)
))

)
