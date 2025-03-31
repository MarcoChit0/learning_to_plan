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
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	infrared0 - mode
	groundstation0 - direction
	star2 - direction
	groundstation5 - direction
	star4 - direction
	star1 - direction
	star3 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 star4)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 star4)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star3)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star4)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star3)
	(calibration_target instrument7 star1)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet7)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite2 star4)
	(have_image star6 infrared0)
	(have_image planet7 infrared0)
	(have_image phenomenon8 infrared0)
))

)
