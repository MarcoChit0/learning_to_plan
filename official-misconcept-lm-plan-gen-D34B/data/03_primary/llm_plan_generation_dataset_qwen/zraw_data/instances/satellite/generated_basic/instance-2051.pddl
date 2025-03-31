(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	satellite5 - satellite
	instrument11 - instrument
	image0 - mode
	groundstation0 - direction
	groundstation2 - direction
	star1 - direction
	star3 - direction
	star4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image0)
	(calibration_target instrument2 star3)
	(supports instrument3 image0)
	(calibration_target instrument3 star3)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument7 image0)
	(calibration_target instrument7 star3)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation2)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument10 image0)
	(calibration_target instrument10 star1)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star4)
	(supports instrument11 image0)
	(calibration_target instrument11 star3)
	(on_board instrument11 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite2 star3)
	(pointing satellite3 groundstation0)
	(pointing satellite4 star3)
	(have_image star4 image0)
))

)
