(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	satellite4 - satellite
	instrument6 - instrument
	satellite5 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image0 - mode
	star1 - direction
	star3 - direction
	star2 - direction
	groundstation0 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument3 image0)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument6 image0)
	(calibration_target instrument6 star2)
	(on_board instrument6 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(calibration_target instrument8 star2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 image0)
	(calibration_target instrument10 star2)
	(supports instrument11 image0)
	(calibration_target instrument11 star2)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation0)
	(on_board instrument7 satellite5)
	(on_board instrument8 satellite5)
	(on_board instrument9 satellite5)
	(on_board instrument10 satellite5)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star3)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite2 planet4)
	(pointing satellite3 star3)
	(pointing satellite5 groundstation0)
	(have_image planet4 image0)
))

)
