(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
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
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	infrared0 - mode
	image3 - mode
	thermograph2 - mode
	thermograph1 - mode
	groundstation3 - direction
	groundstation0 - direction
	groundstation2 - direction
	star1 - direction
	phenomenon4 - direction
	star5 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star1)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 thermograph2)
	(supports instrument7 thermograph1)
	(supports instrument7 image3)
	(calibration_target instrument7 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 infrared0)
	(supports instrument10 thermograph1)
	(supports instrument10 image3)
	(calibration_target instrument10 groundstation3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
	(supports instrument11 thermograph2)
	(supports instrument11 image3)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 thermograph2)
	(supports instrument12 image3)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph1)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 image3)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument16 thermograph1)
	(supports instrument16 infrared0)
	(supports instrument16 thermograph2)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 thermograph1)
	(supports instrument17 infrared0)
	(supports instrument17 image3)
	(calibration_target instrument17 groundstation2)
	(supports instrument18 image3)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star1)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(pointing satellite2 phenomenon4)
	(pointing satellite4 groundstation3)
	(have_image phenomenon4 thermograph2)
	(have_image star5 infrared0)
))

)
