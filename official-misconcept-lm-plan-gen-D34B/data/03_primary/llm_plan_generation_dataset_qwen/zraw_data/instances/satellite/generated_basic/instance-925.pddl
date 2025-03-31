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
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite5 - satellite
	instrument11 - instrument
	instrument12 - instrument
	image1 - mode
	image0 - mode
	star1 - direction
	star2 - direction
	star3 - direction
	groundstation0 - direction
	phenomenon4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 star1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument4 image1)
	(calibration_target instrument4 star3)
	(supports instrument5 image1)
	(calibration_target instrument5 star2)
	(supports instrument6 image1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument8 image0)
	(calibration_target instrument8 star3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon4)
	(supports instrument9 image0)
	(supports instrument9 image1)
	(calibration_target instrument9 star1)
	(supports instrument10 image1)
	(supports instrument10 image0)
	(calibration_target instrument10 star2)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
	(supports instrument11 image1)
	(calibration_target instrument11 star3)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation0)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star3)
)
(:goal (and
	(pointing satellite2 star2)
	(pointing satellite5 groundstation0)
	(have_image phenomenon4 image1)
	(have_image phenomenon5 image1)
))

)
