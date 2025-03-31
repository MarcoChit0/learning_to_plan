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
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	thermograph2 - mode
	spectrograph1 - mode
	infrared0 - mode
	star0 - direction
	groundstation3 - direction
	star1 - direction
	star2 - direction
	star4 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 spectrograph1)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph1)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph1)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation3)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star2)
	(supports instrument9 spectrograph1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 spectrograph1)
	(supports instrument10 thermograph2)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star0)
	(supports instrument11 thermograph2)
	(supports instrument11 spectrograph1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 spectrograph1)
	(supports instrument13 infrared0)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 thermograph2)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 thermograph2)
	(calibration_target instrument15 star1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star2)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 star5)
	(pointing satellite3 star4)
	(have_image star4 infrared0)
	(have_image star5 infrared0)
	(have_image phenomenon6 infrared0)
))

)
