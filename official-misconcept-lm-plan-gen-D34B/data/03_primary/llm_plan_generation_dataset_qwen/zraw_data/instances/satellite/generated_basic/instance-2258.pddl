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
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	image3 - mode
	spectrograph0 - mode
	thermograph2 - mode
	thermograph1 - mode
	star0 - direction
	phenomenon1 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image3)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 image3)
	(supports instrument2 spectrograph0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star0)
	(supports instrument3 image3)
	(calibration_target instrument3 star0)
	(supports instrument4 image3)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon1)
	(supports instrument5 thermograph2)
	(supports instrument5 image3)
	(calibration_target instrument5 star0)
	(supports instrument6 image3)
	(calibration_target instrument6 star0)
	(supports instrument7 image3)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph1)
	(supports instrument9 thermograph2)
	(supports instrument9 image3)
	(calibration_target instrument9 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star0)
	(supports instrument11 thermograph1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph1)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star0)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 star0)
	(supports instrument14 thermograph1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star0)
	(supports instrument15 image3)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star0)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite3 phenomenon1)
	(have_image phenomenon1 spectrograph0)
	(have_image phenomenon2 spectrograph0)
	(have_image planet3 thermograph2)
))

)
