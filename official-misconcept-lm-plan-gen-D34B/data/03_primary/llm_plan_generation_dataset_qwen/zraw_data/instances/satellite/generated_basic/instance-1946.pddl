(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite5 - satellite
	instrument11 - instrument
	instrument12 - instrument
	image0 - mode
	star1 - direction
	star2 - direction
	groundstation0 - direction
	groundstation3 - direction
	groundstation4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation4)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation4)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation3)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon5)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation4)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star6)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite3 star6)
	(have_image phenomenon5 image0)
	(have_image star6 image0)
))

)
