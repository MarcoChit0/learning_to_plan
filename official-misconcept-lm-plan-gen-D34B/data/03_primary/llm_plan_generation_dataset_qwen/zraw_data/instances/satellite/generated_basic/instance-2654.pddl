(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	infrared0 - mode
	groundstation0 - direction
	star2 - direction
	star1 - direction
	star3 - direction
	star5 - direction
	star4 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star5)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star5)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 star1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star5)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star4)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(pointing satellite1 phenomenon6)
	(have_image phenomenon6 infrared0)
))

)
