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
	instrument6 - instrument
	instrument7 - instrument
	infrared1 - mode
	image0 - mode
	star0 - direction
	planet1 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet1)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared1)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite0 planet1)
	(pointing satellite1 planet1)
	(have_image planet1 image0)
))

)
