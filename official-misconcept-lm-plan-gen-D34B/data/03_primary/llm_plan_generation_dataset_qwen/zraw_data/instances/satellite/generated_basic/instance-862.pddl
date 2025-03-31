(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	satellite5 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	thermograph2 - mode
	spectrograph0 - mode
	thermograph1 - mode
	star0 - direction
	star3 - direction
	groundstation2 - direction
	star1 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph1)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star3)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star3)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument7 thermograph2)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument9 thermograph1)
	(supports instrument9 spectrograph0)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation2)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet5)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star3)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 spectrograph0)
	(supports instrument12 thermograph1)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 star1)
	(on_board instrument10 satellite5)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet5)
)
(:goal (and
	(pointing satellite0 planet5)
	(have_image star4 thermograph2)
	(have_image planet5 thermograph1)
))

)
