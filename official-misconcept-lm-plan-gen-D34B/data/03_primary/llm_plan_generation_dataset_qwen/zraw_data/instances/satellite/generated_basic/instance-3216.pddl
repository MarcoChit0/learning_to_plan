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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	image3 - mode
	thermograph1 - mode
	spectrograph2 - mode
	image0 - mode
	star0 - direction
	planet1 - direction
	star2 - direction
	star3 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star0)
	(supports instrument1 image3)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph2)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 spectrograph2)
	(supports instrument4 image0)
	(supports instrument4 image3)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph2)
	(supports instrument5 image3)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph2)
	(supports instrument7 image3)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument8 spectrograph2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star0)
	(supports instrument10 image0)
	(supports instrument10 image3)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star0)
	(supports instrument11 thermograph1)
	(supports instrument11 image3)
	(calibration_target instrument11 star0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument12 image0)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star0)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite1 planet1)
	(pointing satellite2 star2)
	(have_image planet1 image3)
	(have_image star2 thermograph1)
	(have_image star3 thermograph1)
))

)
