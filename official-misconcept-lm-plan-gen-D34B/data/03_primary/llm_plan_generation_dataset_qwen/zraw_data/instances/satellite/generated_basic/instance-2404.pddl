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
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image0 - mode
	thermograph2 - mode
	thermograph3 - mode
	spectrograph1 - mode
	groundstation3 - direction
	groundstation1 - direction
	groundstation4 - direction
	star2 - direction
	star0 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph3)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 image0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph3)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 image0)
	(supports instrument6 thermograph2)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph1)
	(supports instrument7 thermograph3)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph1)
	(supports instrument8 thermograph3)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 image0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument10 spectrograph1)
	(supports instrument10 thermograph3)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 groundstation4)
	(supports instrument11 spectrograph1)
	(supports instrument11 thermograph3)
	(calibration_target instrument11 star2)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(pointing satellite2 groundstation4)
	(have_image planet5 thermograph3)
	(have_image star6 image0)
))

)
