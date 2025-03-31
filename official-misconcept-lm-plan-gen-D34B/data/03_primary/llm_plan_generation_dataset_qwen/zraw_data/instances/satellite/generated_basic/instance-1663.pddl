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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	infrared0 - mode
	star0 - direction
	groundstation3 - direction
	star2 - direction
	star5 - direction
	groundstation1 - direction
	groundstation4 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation3)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star5)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 star2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation4)
	(calibration_target instrument8 groundstation1)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(pointing satellite2 groundstation4)
	(have_image phenomenon6 infrared0)
))

)
