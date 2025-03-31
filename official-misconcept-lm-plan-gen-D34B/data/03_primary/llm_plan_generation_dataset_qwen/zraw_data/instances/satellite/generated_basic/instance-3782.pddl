(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	satellite5 - satellite
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	instrument23 - instrument
	infrared0 - mode
	star5 - direction
	groundstation2 - direction
	groundstation4 - direction
	star3 - direction
	groundstation1 - direction
	star0 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation4)
	(calibration_target instrument1 star5)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star5)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation4)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star5)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation2)
	(calibration_target instrument10 groundstation4)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star0)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation1)
	(calibration_target instrument13 star3)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation2)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star3)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 infrared0)
	(calibration_target instrument17 groundstation4)
	(calibration_target instrument17 star5)
	(supports instrument18 infrared0)
	(calibration_target instrument18 groundstation1)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star6)
	(supports instrument19 infrared0)
	(calibration_target instrument19 groundstation2)
	(supports instrument20 infrared0)
	(calibration_target instrument20 star3)
	(calibration_target instrument20 groundstation4)
	(supports instrument21 infrared0)
	(calibration_target instrument21 star0)
	(supports instrument22 infrared0)
	(calibration_target instrument22 groundstation1)
	(supports instrument23 infrared0)
	(calibration_target instrument23 star0)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation4)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(have_image star6 infrared0)
))

)
