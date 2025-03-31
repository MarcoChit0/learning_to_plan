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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	spectrograph0 - mode
	thermograph2 - mode
	thermograph1 - mode
	groundstation3 - direction
	star0 - direction
	groundstation1 - direction
	star2 - direction
	planet4 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument5 thermograph1)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph1)
	(supports instrument7 thermograph2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph0)
	(supports instrument9 thermograph1)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 thermograph2)
	(supports instrument11 thermograph1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument12 spectrograph0)
	(supports instrument12 thermograph2)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star2)
	(supports instrument13 thermograph2)
	(supports instrument13 spectrograph0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 star2)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 groundstation3)
	(pointing satellite2 star2)
	(pointing satellite3 star2)
	(have_image planet4 spectrograph0)
))

)
