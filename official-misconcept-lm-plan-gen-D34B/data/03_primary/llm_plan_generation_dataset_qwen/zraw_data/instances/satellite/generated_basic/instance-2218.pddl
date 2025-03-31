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
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
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
	infrared0 - mode
	image1 - mode
	groundstation2 - direction
	star1 - direction
	star0 - direction
	planet3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument6 image1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star1)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
	(supports instrument11 image1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 image1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star0)
	(supports instrument15 infrared0)
	(calibration_target instrument15 star0)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument17 image1)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star1)
	(supports instrument18 image1)
	(supports instrument18 infrared0)
	(calibration_target instrument18 groundstation2)
	(supports instrument19 image1)
	(supports instrument19 infrared0)
	(calibration_target instrument19 star1)
	(supports instrument20 infrared0)
	(supports instrument20 image1)
	(calibration_target instrument20 groundstation2)
	(supports instrument21 image1)
	(supports instrument21 infrared0)
	(calibration_target instrument21 star1)
	(supports instrument22 image1)
	(supports instrument22 infrared0)
	(calibration_target instrument22 star0)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(on_board instrument22 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon4)
)
(:goal (and
	(pointing satellite1 star1)
	(pointing satellite2 groundstation2)
	(have_image planet3 infrared0)
	(have_image phenomenon4 image1)
))

)
