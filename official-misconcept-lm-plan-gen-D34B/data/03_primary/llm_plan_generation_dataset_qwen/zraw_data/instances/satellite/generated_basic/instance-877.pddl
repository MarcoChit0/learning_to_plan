(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	thermograph2 - mode
	spectrograph1 - mode
	infrared0 - mode
	star2 - direction
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	phenomenon4 - direction
	star5 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph1)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star3)
	(supports instrument9 infrared0)
	(supports instrument9 spectrograph1)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star2)
	(supports instrument10 spectrograph1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument11 infrared0)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 thermograph2)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star2)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument14 thermograph2)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 infrared0)
	(supports instrument15 thermograph2)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 groundstation0)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
	(supports instrument16 thermograph2)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 spectrograph1)
	(supports instrument17 thermograph2)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star3)
	(supports instrument18 spectrograph1)
	(supports instrument18 infrared0)
	(supports instrument18 thermograph2)
	(calibration_target instrument18 star3)
	(supports instrument19 thermograph2)
	(calibration_target instrument19 star3)
	(supports instrument20 thermograph2)
	(supports instrument20 infrared0)
	(calibration_target instrument20 groundstation1)
	(supports instrument21 thermograph2)
	(supports instrument21 infrared0)
	(supports instrument21 spectrograph1)
	(calibration_target instrument21 star3)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star5)
)
(:goal (and
	(pointing satellite0 phenomenon4)
	(pointing satellite1 star5)
	(pointing satellite4 phenomenon4)
	(pointing satellite5 star5)
	(have_image phenomenon4 spectrograph1)
	(have_image star5 infrared0)
))

)
