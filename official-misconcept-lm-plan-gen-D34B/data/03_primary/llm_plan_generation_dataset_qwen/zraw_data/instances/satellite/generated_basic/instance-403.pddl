(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	image0 - mode
	image1 - mode
	groundstation0 - direction
	planet1 - direction
	star2 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet1)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument5 image1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument6 image1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet1)
)
(:goal (and
	(pointing satellite0 planet1)
	(pointing satellite3 planet1)
	(have_image planet1 image0)
	(have_image star2 image1)
))

)
