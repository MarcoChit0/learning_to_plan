(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	image0 - mode
	image1 - mode
	star0 - direction
	phenomenon1 - direction
	planet2 - direction
	star3 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
)
(:goal (and
	(pointing satellite0 star4)
	(have_image phenomenon1 image1)
	(have_image planet2 image0)
	(have_image star3 image1)
	(have_image star4 image1)
	(have_image planet5 image1)
))

)
