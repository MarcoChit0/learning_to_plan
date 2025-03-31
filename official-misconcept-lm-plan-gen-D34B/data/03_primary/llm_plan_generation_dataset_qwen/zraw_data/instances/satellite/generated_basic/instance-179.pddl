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
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite4 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite5 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	image0 - mode
	star1 - direction
	star0 - direction
	planet2 - direction
	phenomenon3 - direction
	star4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(calibration_target instrument5 star1)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(supports instrument8 image0)
	(calibration_target instrument8 star1)
	(supports instrument9 image0)
	(calibration_target instrument9 star1)
	(on_board instrument7 satellite4)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet2)
	(supports instrument10 image0)
	(calibration_target instrument10 star1)
	(supports instrument11 image0)
	(calibration_target instrument11 star1)
	(supports instrument12 image0)
	(calibration_target instrument12 star1)
	(supports instrument13 image0)
	(calibration_target instrument13 star0)
	(on_board instrument10 satellite5)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite3 star1)
	(have_image planet2 image0)
	(have_image phenomenon3 image0)
	(have_image star4 image0)
))

)
