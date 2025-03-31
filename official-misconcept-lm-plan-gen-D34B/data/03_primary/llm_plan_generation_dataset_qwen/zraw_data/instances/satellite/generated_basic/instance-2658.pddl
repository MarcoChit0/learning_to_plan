(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	infrared0 - mode
	groundstation4 - direction
	star3 - direction
	star0 - direction
	groundstation1 - direction
	star2 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star5)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(calibration_target instrument3 star3)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star5)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 groundstation1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star5)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite2 star0)
	(pointing satellite3 phenomenon6)
	(have_image phenomenon6 infrared0)
))

)
