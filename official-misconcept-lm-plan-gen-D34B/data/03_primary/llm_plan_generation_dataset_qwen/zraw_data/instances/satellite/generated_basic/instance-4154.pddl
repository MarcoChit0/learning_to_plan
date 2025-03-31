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
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	image0 - mode
	thermograph1 - mode
	thermograph3 - mode
	spectrograph2 - mode
	groundstation1 - direction
	star0 - direction
	planet2 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph1)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph3)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 spectrograph2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument10 thermograph3)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star0)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 thermograph3)
	(supports instrument12 image0)
	(calibration_target instrument12 star0)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 star0)
	(supports instrument14 thermograph3)
	(supports instrument14 image0)
	(calibration_target instrument14 star0)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument15 image0)
	(supports instrument15 spectrograph2)
	(supports instrument15 thermograph3)
	(calibration_target instrument15 star0)
	(supports instrument16 thermograph3)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 star0)
	(supports instrument17 thermograph3)
	(calibration_target instrument17 star0)
	(supports instrument18 spectrograph2)
	(supports instrument18 thermograph1)
	(calibration_target instrument18 star0)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(pointing satellite3 planet2)
	(have_image planet2 image0)
))

)
