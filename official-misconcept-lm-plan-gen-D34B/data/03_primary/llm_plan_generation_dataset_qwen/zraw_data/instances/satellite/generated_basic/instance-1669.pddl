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
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	infrared0 - mode
	groundstation5 - direction
	groundstation0 - direction
	groundstation4 - direction
	star2 - direction
	groundstation1 - direction
	groundstation3 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation4)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation3)
	(calibration_target instrument9 star2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation4)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation1)
	(calibration_target instrument12 groundstation4)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation4)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star2)
	(calibration_target instrument14 groundstation1)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation4)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star2)
	(calibration_target instrument16 groundstation3)
	(supports instrument17 infrared0)
	(calibration_target instrument17 groundstation3)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star2)
	(supports instrument19 infrared0)
	(calibration_target instrument19 groundstation3)
	(calibration_target instrument19 groundstation1)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(on_board instrument19 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
)
(:goal (and
	(pointing satellite2 groundstation5)
	(pointing satellite3 groundstation3)
	(have_image phenomenon6 infrared0)
))

)
