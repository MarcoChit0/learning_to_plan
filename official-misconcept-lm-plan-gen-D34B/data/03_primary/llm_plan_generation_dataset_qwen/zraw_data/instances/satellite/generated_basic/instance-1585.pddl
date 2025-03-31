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
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	satellite5 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	infrared0 - mode
	star0 - direction
	groundstation3 - direction
	groundstation1 - direction
	groundstation2 - direction
	groundstation4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation1)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star0)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation3)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 infrared0)
	(calibration_target instrument17 groundstation4)
	(supports instrument18 infrared0)
	(calibration_target instrument18 groundstation4)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation3)
)
(:goal (and
	(pointing satellite2 star0)
	(pointing satellite4 star7)
	(have_image phenomenon5 infrared0)
	(have_image phenomenon6 infrared0)
	(have_image star7 infrared0)
))

)
