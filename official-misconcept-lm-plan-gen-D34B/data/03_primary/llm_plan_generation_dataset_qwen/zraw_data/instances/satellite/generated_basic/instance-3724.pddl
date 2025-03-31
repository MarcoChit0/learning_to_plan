(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
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
	instrument11 - instrument
	instrument12 - instrument
	infrared1 - mode
	spectrograph0 - mode
	thermograph2 - mode
	star0 - direction
	star4 - direction
	groundstation3 - direction
	star2 - direction
	star5 - direction
	groundstation1 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star4)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 thermograph2)
	(supports instrument6 spectrograph0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star2)
	(calibration_target instrument6 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star2)
	(calibration_target instrument7 star4)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 groundstation3)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation3)
	(calibration_target instrument9 star4)
	(supports instrument10 thermograph2)
	(supports instrument10 spectrograph0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star2)
	(supports instrument11 infrared1)
	(supports instrument11 thermograph2)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 star5)
	(supports instrument12 infrared1)
	(supports instrument12 spectrograph0)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite1 groundstation1)
	(have_image star6 spectrograph0)
))

)
