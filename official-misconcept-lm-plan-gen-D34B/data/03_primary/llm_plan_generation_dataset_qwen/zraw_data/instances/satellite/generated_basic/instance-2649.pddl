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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	infrared0 - mode
	groundstation2 - direction
	groundstation5 - direction
	groundstation4 - direction
	star1 - direction
	star0 - direction
	groundstation3 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 groundstation5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation4)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation4)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star1)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation3)
	(calibration_target instrument12 star0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(pointing satellite3 star0)
	(have_image star6 infrared0)
))

)
