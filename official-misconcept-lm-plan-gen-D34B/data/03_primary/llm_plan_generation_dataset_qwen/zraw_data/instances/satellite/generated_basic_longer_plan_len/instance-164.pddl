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
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	satellite6 - satellite
	instrument21 - instrument
	image0 - mode
	image1 - mode
	groundstation0 - direction
	phenomenon1 - direction
	planet2 - direction
	planet3 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet2)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image0)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image1)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation0)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 image0)
	(supports instrument17 image1)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 image0)
	(supports instrument18 image1)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 image0)
	(supports instrument19 image1)
	(calibration_target instrument19 groundstation0)
	(supports instrument20 image0)
	(supports instrument20 image1)
	(calibration_target instrument20 groundstation0)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
	(supports instrument21 image0)
	(supports instrument21 image1)
	(calibration_target instrument21 groundstation0)
	(on_board instrument21 satellite6)
	(power_avail satellite6)
	(pointing satellite6 groundstation0)
)
(:goal (and
	(pointing satellite1 planet2)
	(pointing satellite2 planet6)
	(pointing satellite3 planet2)
	(pointing satellite4 planet6)
	(have_image phenomenon1 image1)
	(have_image planet2 image1)
	(have_image planet3 image0)
	(have_image star4 image0)
	(have_image planet5 image1)
	(have_image planet6 image0)
	(have_image phenomenon7 image1)
	(have_image phenomenon8 image0)
))

)
