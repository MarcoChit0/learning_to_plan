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
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image0 - mode
	image1 - mode
	groundstation1 - direction
	groundstation3 - direction
	groundstation4 - direction
	star0 - direction
	groundstation6 - direction
	star2 - direction
	star5 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	star9 - direction
	planet10 - direction
	star11 - direction
	star12 - direction
	star13 - direction
	phenomenon14 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation4)
	(calibration_target instrument1 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star13)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation6)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation3)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 image0)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(supports instrument7 image1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet10)
	(supports instrument8 image0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation4)
	(calibration_target instrument8 star5)
	(supports instrument9 image1)
	(supports instrument9 image0)
	(calibration_target instrument9 star5)
	(calibration_target instrument9 star2)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation6)
	(calibration_target instrument10 star0)
	(supports instrument11 image1)
	(calibration_target instrument11 star2)
	(supports instrument12 image0)
	(supports instrument12 image1)
	(calibration_target instrument12 star5)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet10)
)
(:goal (and
	(pointing satellite2 groundstation6)
	(have_image phenomenon7 image0)
	(have_image phenomenon8 image0)
	(have_image star9 image0)
	(have_image planet10 image0)
	(have_image star11 image1)
	(have_image star12 image1)
	(have_image star13 image1)
	(have_image phenomenon14 image0)
))

)
