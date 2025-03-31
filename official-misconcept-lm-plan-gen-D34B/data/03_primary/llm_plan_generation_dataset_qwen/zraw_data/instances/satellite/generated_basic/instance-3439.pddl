(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
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
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	satellite5 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	image0 - mode
	image2 - mode
	image1 - mode
	groundstation1 - direction
	groundstation0 - direction
	planet2 - direction
	phenomenon3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument5 image0)
	(supports instrument5 image2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image2)
	(supports instrument7 image1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image0)
	(supports instrument9 image1)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument10 image1)
	(supports instrument10 image2)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image0)
	(supports instrument11 image2)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image2)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon3)
	(supports instrument14 image2)
	(supports instrument14 image0)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation1)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon4)
	(supports instrument15 image2)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 image0)
	(supports instrument16 image2)
	(supports instrument16 image1)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 image0)
	(supports instrument17 image1)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 image1)
	(supports instrument18 image0)
	(supports instrument18 image2)
	(calibration_target instrument18 groundstation0)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon4)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite4 planet2)
	(have_image planet2 image0)
	(have_image phenomenon3 image0)
	(have_image phenomenon4 image1)
))

)
