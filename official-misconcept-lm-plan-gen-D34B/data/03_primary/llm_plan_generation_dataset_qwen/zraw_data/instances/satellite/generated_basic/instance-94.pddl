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
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	image1 - mode
	image0 - mode
	star3 - direction
	groundstation2 - direction
	groundstation0 - direction
	star4 - direction
	star1 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image1)
	(calibration_target instrument3 star3)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 star4)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image1)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(calibration_target instrument7 star5)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument8 image0)
	(supports instrument8 image1)
	(calibration_target instrument8 star5)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation2)
	(calibration_target instrument9 groundstation0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star6)
	(supports instrument10 image1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 image0)
	(supports instrument12 image1)
	(calibration_target instrument12 star5)
	(calibration_target instrument12 star4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet9)
	(supports instrument13 image1)
	(supports instrument13 image0)
	(calibration_target instrument13 star4)
	(supports instrument14 image0)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 image0)
	(supports instrument15 image1)
	(calibration_target instrument15 star4)
	(calibration_target instrument15 star1)
	(supports instrument16 image1)
	(calibration_target instrument16 star5)
	(calibration_target instrument16 star1)
	(supports instrument17 image1)
	(calibration_target instrument17 star5)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
)
(:goal (and
	(pointing satellite3 phenomenon7)
	(have_image star6 image1)
	(have_image phenomenon7 image0)
	(have_image planet8 image0)
	(have_image planet9 image1)
))

)
