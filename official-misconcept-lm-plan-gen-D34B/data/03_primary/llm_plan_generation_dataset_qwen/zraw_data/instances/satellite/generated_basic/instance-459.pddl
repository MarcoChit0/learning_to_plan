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
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	image0 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	star3 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(supports instrument3 image0)
	(calibration_target instrument3 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
)
(:goal (and
	(have_image star2 image0)
	(have_image star3 image0)
))

)
