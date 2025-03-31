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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	image1 - mode
	image0 - mode
	groundstation0 - direction
	groundstation2 - direction
	star1 - direction
	star3 - direction
	star4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(supports instrument6 image0)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image1)
	(supports instrument8 image0)
	(calibration_target instrument8 star1)
	(supports instrument9 image1)
	(calibration_target instrument9 star1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
)
(:goal (and
	(have_image star3 image0)
	(have_image star4 image1)
	(have_image phenomenon5 image1)
	(have_image phenomenon6 image0)
	(have_image star7 image0)
))

)
