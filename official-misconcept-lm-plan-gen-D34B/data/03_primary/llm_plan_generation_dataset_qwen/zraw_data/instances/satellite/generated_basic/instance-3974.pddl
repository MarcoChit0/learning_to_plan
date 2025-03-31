(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	image1 - mode
	infrared0 - mode
	star0 - direction
	planet1 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet1)
	(supports instrument3 infrared0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet1)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared0)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet1)
)
(:goal (and
	(pointing satellite1 planet1)
	(have_image planet1 infrared0)
))

)
