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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	image0 - mode
	star5 - direction
	star2 - direction
	groundstation1 - direction
	star4 - direction
	groundstation0 - direction
	groundstation3 - direction
	star6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 star2)
	(supports instrument1 image0)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 image0)
	(calibration_target instrument2 star5)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 star5)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(calibration_target instrument6 star2)
	(supports instrument7 image0)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(calibration_target instrument8 star2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(calibration_target instrument9 groundstation0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image0)
	(calibration_target instrument11 star4)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation0)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation3)
	(calibration_target instrument14 groundstation0)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star6)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 groundstation0)
	(have_image star6 image0)
	(have_image planet7 image0)
	(have_image star8 image0)
))

)
