(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	spectrograph1 - mode
	thermograph2 - mode
	thermograph0 - mode
	star1 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation0 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph1)
	(supports instrument2 thermograph2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph0)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument7 spectrograph1)
	(supports instrument7 thermograph2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 thermograph0)
	(supports instrument8 thermograph2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star1)
	(supports instrument12 thermograph0)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star1)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
	(supports instrument13 thermograph2)
	(supports instrument13 thermograph0)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star1)
	(supports instrument14 thermograph2)
	(supports instrument14 thermograph0)
	(supports instrument14 spectrograph1)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 spectrograph1)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation0)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
	(supports instrument16 spectrograph1)
	(supports instrument16 thermograph0)
	(supports instrument16 thermograph2)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 spectrograph1)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 spectrograph1)
	(supports instrument18 thermograph0)
	(supports instrument18 thermograph2)
	(calibration_target instrument18 groundstation3)
	(supports instrument19 spectrograph1)
	(supports instrument19 thermograph2)
	(calibration_target instrument19 groundstation3)
	(supports instrument20 spectrograph1)
	(supports instrument20 thermograph2)
	(calibration_target instrument20 groundstation0)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet4)
)
(:goal (and
	(pointing satellite2 phenomenon5)
	(pointing satellite3 groundstation3)
	(pointing satellite4 planet4)
	(have_image planet4 spectrograph1)
	(have_image phenomenon5 thermograph2)
))

)
