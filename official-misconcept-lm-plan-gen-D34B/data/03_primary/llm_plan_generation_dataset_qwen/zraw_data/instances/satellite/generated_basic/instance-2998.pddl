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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	infrared0 - mode
	groundstation1 - direction
	star0 - direction
	star5 - direction
	star2 - direction
	star3 - direction
	groundstation4 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star5)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star2)
	(calibration_target instrument8 star5)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation4)
	(calibration_target instrument9 star3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 star3)
	(pointing satellite3 star3)
	(have_image phenomenon6 infrared0)
	(have_image phenomenon7 infrared0)
	(have_image planet8 infrared0)
))

)
