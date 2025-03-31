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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	satellite5 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	infrared0 - mode
	groundstation0 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation1 - direction
	groundstation4 - direction
	groundstation5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 groundstation2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation5)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation4)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation5)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation4)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation4)
	(calibration_target instrument13 groundstation3)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation4)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation5)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation5)
	(pointing satellite2 groundstation0)
	(have_image star6 infrared0)
))

)
