(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	image1 - mode
	image0 - mode
	star0 - direction
	star3 - direction
	groundstation1 - direction
	star4 - direction
	star2 - direction
	phenomenon5 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 image0)
	(calibration_target instrument2 star4)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument6 image0)
	(calibration_target instrument6 star2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument7 image0)
	(supports instrument7 image1)
	(calibration_target instrument7 star4)
	(supports instrument8 image0)
	(supports instrument8 image1)
	(calibration_target instrument8 star4)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation1)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image1)
	(supports instrument12 image0)
	(calibration_target instrument12 star4)
	(supports instrument13 image1)
	(supports instrument13 image0)
	(calibration_target instrument13 star2)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
)
(:goal (and
	(pointing satellite2 phenomenon5)
	(pointing satellite4 star0)
	(have_image phenomenon5 image0)
	(have_image planet6 image1)
	(have_image star7 image1)
	(have_image planet8 image0)
))

)
