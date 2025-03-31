(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	satellite5 - satellite
	instrument13 - instrument
	instrument14 - instrument
	thermograph0 - mode
	spectrograph2 - mode
	thermograph1 - mode
	groundstation2 - direction
	star1 - direction
	groundstation3 - direction
	star0 - direction
	planet4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 thermograph0)
	(supports instrument5 spectrograph2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph1)
	(supports instrument9 spectrograph2)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument11 thermograph0)
	(supports instrument11 spectrograph2)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 star1)
	(supports instrument12 thermograph0)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star1)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet5)
	(supports instrument13 spectrograph2)
	(supports instrument13 thermograph0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 thermograph0)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 star0)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star1)
)
(:goal (and
	(pointing satellite3 planet4)
	(pointing satellite4 planet5)
	(have_image planet4 thermograph0)
	(have_image planet5 thermograph0)
))

)
