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
	satellite5 - satellite
	instrument22 - instrument
	instrument23 - instrument
	instrument24 - instrument
	instrument25 - instrument
	infrared0 - mode
	groundstation5 - direction
	groundstation4 - direction
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	groundstation2 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 star3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation0)
	(calibration_target instrument8 groundstation5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation2)
	(calibration_target instrument9 star3)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation2)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation0)
	(calibration_target instrument11 groundstation2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star6)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star3)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star3)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation4)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation4)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation4)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star3)
	(supports instrument18 infrared0)
	(calibration_target instrument18 groundstation1)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 infrared0)
	(calibration_target instrument19 star3)
	(calibration_target instrument19 groundstation5)
	(supports instrument20 infrared0)
	(calibration_target instrument20 groundstation0)
	(supports instrument21 infrared0)
	(calibration_target instrument21 groundstation4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
	(supports instrument22 infrared0)
	(calibration_target instrument22 groundstation2)
	(supports instrument23 infrared0)
	(calibration_target instrument23 groundstation0)
	(supports instrument24 infrared0)
	(calibration_target instrument24 groundstation1)
	(calibration_target instrument24 star3)
	(supports instrument25 infrared0)
	(calibration_target instrument25 groundstation2)
	(calibration_target instrument25 star3)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(on_board instrument24 satellite5)
	(on_board instrument25 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation4)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 groundstation2)
	(pointing satellite4 star6)
	(pointing satellite5 star3)
	(have_image star6 infrared0)
))

)
