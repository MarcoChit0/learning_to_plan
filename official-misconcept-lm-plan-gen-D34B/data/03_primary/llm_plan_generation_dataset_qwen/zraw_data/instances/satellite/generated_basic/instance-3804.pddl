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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	satellite5 - satellite
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	infrared0 - mode
	spectrograph3 - mode
	thermograph2 - mode
	thermograph1 - mode
	groundstation1 - direction
	star3 - direction
	groundstation2 - direction
	star0 - direction
	star4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 star3)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared0)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 spectrograph3)
	(supports instrument6 infrared0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument7 thermograph2)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 thermograph2)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared0)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star3)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument12 spectrograph3)
	(supports instrument12 infrared0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 thermograph1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star0)
	(supports instrument14 spectrograph3)
	(supports instrument14 thermograph1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 thermograph2)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 groundstation2)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
	(supports instrument16 thermograph2)
	(calibration_target instrument16 star3)
	(supports instrument17 thermograph1)
	(supports instrument17 thermograph2)
	(calibration_target instrument17 star3)
	(supports instrument18 infrared0)
	(supports instrument18 thermograph1)
	(calibration_target instrument18 star0)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
	(supports instrument19 thermograph1)
	(supports instrument19 thermograph2)
	(supports instrument19 spectrograph3)
	(calibration_target instrument19 star3)
	(supports instrument20 spectrograph3)
	(supports instrument20 thermograph2)
	(calibration_target instrument20 star0)
	(supports instrument21 thermograph1)
	(supports instrument21 spectrograph3)
	(calibration_target instrument21 groundstation2)
	(supports instrument22 thermograph1)
	(calibration_target instrument22 star0)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite2 star3)
	(have_image star4 infrared0)
	(have_image phenomenon5 thermograph1)
))

)
