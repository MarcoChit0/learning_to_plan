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
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	infrared0 - mode
	image1 - mode
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	star0 - direction
	phenomenon4 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument1 infrared0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared0)
	(supports instrument4 image1)
	(calibration_target instrument4 star2)
	(supports instrument5 image1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star2)
	(supports instrument8 infrared0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 infrared0)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument12 infrared0)
	(supports instrument12 image1)
	(calibration_target instrument12 star2)
	(supports instrument13 infrared0)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 infrared0)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 image1)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument16 infrared0)
	(supports instrument16 image1)
	(calibration_target instrument16 star2)
	(supports instrument17 infrared0)
	(calibration_target instrument17 groundstation3)
	(supports instrument18 image1)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star0)
	(supports instrument19 image1)
	(calibration_target instrument19 groundstation1)
	(supports instrument20 infrared0)
	(supports instrument20 image1)
	(calibration_target instrument20 groundstation1)
	(supports instrument21 image1)
	(supports instrument21 infrared0)
	(calibration_target instrument21 star0)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation3)
)
(:goal (and
	(pointing satellite1 star0)
	(pointing satellite2 groundstation3)
	(have_image phenomenon4 image1)
	(have_image planet5 image1)
	(have_image star6 image1)
))

)
