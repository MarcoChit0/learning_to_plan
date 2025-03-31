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
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite4 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	image1 - mode
	infrared0 - mode
	thermograph2 - mode
	spectrograph3 - mode
	star1 - direction
	star0 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument5 spectrograph3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared0)
	(supports instrument7 image1)
	(calibration_target instrument7 star1)
	(supports instrument8 spectrograph3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument9 spectrograph3)
	(supports instrument9 image1)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph2)
	(supports instrument10 spectrograph3)
	(calibration_target instrument10 star0)
	(supports instrument11 image1)
	(supports instrument11 spectrograph3)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star0)
	(supports instrument13 thermograph2)
	(supports instrument13 image1)
	(supports instrument13 spectrograph3)
	(calibration_target instrument13 star0)
	(supports instrument14 spectrograph3)
	(supports instrument14 image1)
	(calibration_target instrument14 star0)
	(supports instrument15 infrared0)
	(supports instrument15 spectrograph3)
	(calibration_target instrument15 star1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star0)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet3)
	(supports instrument17 thermograph2)
	(calibration_target instrument17 star1)
	(supports instrument18 thermograph2)
	(calibration_target instrument18 star1)
	(supports instrument19 spectrograph3)
	(supports instrument19 image1)
	(calibration_target instrument19 star1)
	(supports instrument20 image1)
	(supports instrument20 thermograph2)
	(calibration_target instrument20 star0)
	(supports instrument21 image1)
	(supports instrument21 spectrograph3)
	(supports instrument21 thermograph2)
	(calibration_target instrument21 star1)
	(supports instrument22 infrared0)
	(calibration_target instrument22 star0)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(on_board instrument22 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon2)
)
(:goal (and
	(pointing satellite2 phenomenon2)
	(pointing satellite4 star0)
	(have_image phenomenon2 thermograph2)
	(have_image planet3 thermograph2)
))

)
