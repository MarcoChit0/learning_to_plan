(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	infrared0 - mode
	star0 - direction
	star1 - direction
	groundstation2 - direction
	groundstation4 - direction
	groundstation5 - direction
	groundstation3 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation5)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation3)
	(calibration_target instrument5 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation3)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation5)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation5)
)
(:goal (and
	(pointing satellite1 star1)
	(pointing satellite2 groundstation5)
	(have_image phenomenon6 infrared0)
))

)
