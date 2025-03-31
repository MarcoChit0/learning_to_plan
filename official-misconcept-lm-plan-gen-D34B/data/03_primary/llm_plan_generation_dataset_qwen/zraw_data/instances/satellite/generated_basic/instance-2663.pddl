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
	infrared0 - mode
	star2 - direction
	star0 - direction
	star1 - direction
	groundstation5 - direction
	groundstation4 - direction
	groundstation3 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation5)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation3)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation4)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(pointing satellite1 groundstation3)
	(pointing satellite2 star2)
	(have_image phenomenon6 infrared0)
))

)
