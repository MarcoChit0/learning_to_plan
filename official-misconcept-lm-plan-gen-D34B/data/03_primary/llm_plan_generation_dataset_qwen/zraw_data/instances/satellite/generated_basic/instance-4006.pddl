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
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	spectrograph3 - mode
	image0 - mode
	thermograph2 - mode
	infrared1 - mode
	star1 - direction
	star0 - direction
	planet2 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 spectrograph3)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph2)
	(supports instrument1 image0)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph3)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(supports instrument3 thermograph2)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet2)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph2)
	(supports instrument6 image0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared1)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 star1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument9 infrared1)
	(supports instrument9 spectrograph3)
	(supports instrument9 image0)
	(calibration_target instrument9 star1)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star1)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(supports instrument12 spectrograph3)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star1)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument13 image0)
	(calibration_target instrument13 star0)
	(supports instrument14 image0)
	(supports instrument14 infrared1)
	(supports instrument14 spectrograph3)
	(calibration_target instrument14 star0)
	(supports instrument15 thermograph2)
	(calibration_target instrument15 star1)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet2)
	(supports instrument16 spectrograph3)
	(supports instrument16 infrared1)
	(calibration_target instrument16 star1)
	(supports instrument17 spectrograph3)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star1)
	(supports instrument18 spectrograph3)
	(supports instrument18 infrared1)
	(calibration_target instrument18 star1)
	(supports instrument19 spectrograph3)
	(calibration_target instrument19 star0)
	(supports instrument20 infrared1)
	(calibration_target instrument20 star0)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
)
(:goal (and
	(pointing satellite1 star1)
	(pointing satellite2 star0)
	(have_image planet2 thermograph2)
))

)
