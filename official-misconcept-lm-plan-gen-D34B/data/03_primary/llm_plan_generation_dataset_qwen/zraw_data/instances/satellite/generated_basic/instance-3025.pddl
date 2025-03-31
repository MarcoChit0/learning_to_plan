(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	image0 - mode
	image1 - mode
	star1 - direction
	groundstation2 - direction
	star3 - direction
	groundstation0 - direction
	phenomenon4 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 star3)
	(supports instrument5 image1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(supports instrument7 image1)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(supports instrument8 image1)
	(supports instrument8 image0)
	(calibration_target instrument8 star1)
	(supports instrument9 image1)
	(supports instrument9 image0)
	(calibration_target instrument9 star1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument11 image1)
	(supports instrument11 image0)
	(calibration_target instrument11 star1)
	(supports instrument12 image0)
	(supports instrument12 image1)
	(calibration_target instrument12 star1)
	(supports instrument13 image0)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation2)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon4)
	(supports instrument14 image1)
	(supports instrument14 image0)
	(calibration_target instrument14 star3)
	(supports instrument15 image0)
	(supports instrument15 image1)
	(calibration_target instrument15 star3)
	(supports instrument16 image1)
	(calibration_target instrument16 groundstation0)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
)
(:goal (and
	(have_image phenomenon4 image1)
	(have_image planet5 image0)
	(have_image star6 image0)
))

)
