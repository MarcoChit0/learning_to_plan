(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	image0 - mode
	image1 - mode
	groundstation2 - direction
	groundstation1 - direction
	star3 - direction
	star0 - direction
	star4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image1)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(supports instrument8 image0)
	(supports instrument8 image1)
	(calibration_target instrument8 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument9 image0)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
)
(:goal (and
	(pointing satellite2 planet5)
	(have_image star4 image0)
	(have_image planet5 image1)
	(have_image phenomenon6 image0)
))

)
