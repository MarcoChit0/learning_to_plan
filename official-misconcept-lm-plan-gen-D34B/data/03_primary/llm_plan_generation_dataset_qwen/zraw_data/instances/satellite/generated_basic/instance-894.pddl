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
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	thermograph1 - mode
	spectrograph0 - mode
	spectrograph2 - mode
	star1 - direction
	star3 - direction
	star2 - direction
	star0 - direction
	planet4 - direction
	star5 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph1)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph2)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph1)
	(supports instrument9 spectrograph2)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument10 thermograph1)
	(supports instrument10 spectrograph2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star0)
	(supports instrument11 spectrograph2)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star2)
	(supports instrument12 spectrograph0)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph1)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 star1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 star0)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star1)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument16 spectrograph2)
	(supports instrument16 thermograph1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star0)
	(supports instrument17 spectrograph0)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 star2)
	(supports instrument18 thermograph1)
	(calibration_target instrument18 star3)
	(supports instrument19 spectrograph0)
	(supports instrument19 thermograph1)
	(supports instrument19 spectrograph2)
	(calibration_target instrument19 star2)
	(supports instrument20 spectrograph2)
	(supports instrument20 thermograph1)
	(calibration_target instrument20 star0)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star5)
)
(:goal (and
	(pointing satellite2 star2)
	(pointing satellite4 star0)
	(have_image planet4 spectrograph2)
	(have_image star5 thermograph1)
))

)
