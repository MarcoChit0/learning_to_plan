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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	satellite6 - satellite
	instrument18 - instrument
	instrument19 - instrument
	image0 - mode
	infrared1 - mode
	groundstation0 - direction
	phenomenon1 - direction
	planet2 - direction
	phenomenon3 - direction
	planet4 - direction
	star5 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
	(supports instrument11 image0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 infrared1)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 image0)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 infrared1)
	(calibration_target instrument16 groundstation0)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet4)
	(supports instrument17 image0)
	(supports instrument17 infrared1)
	(calibration_target instrument17 groundstation0)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon3)
	(supports instrument18 image0)
	(supports instrument18 infrared1)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 image0)
	(supports instrument19 infrared1)
	(calibration_target instrument19 groundstation0)
	(on_board instrument18 satellite6)
	(on_board instrument19 satellite6)
	(power_avail satellite6)
	(pointing satellite6 planet6)
)
(:goal (and
	(pointing satellite0 phenomenon1)
	(pointing satellite1 phenomenon3)
	(pointing satellite2 phenomenon8)
	(have_image phenomenon1 infrared1)
	(have_image planet2 infrared1)
	(have_image phenomenon3 image0)
	(have_image planet4 infrared1)
	(have_image star5 image0)
	(have_image planet6 infrared1)
	(have_image star7 infrared1)
	(have_image phenomenon8 infrared1)
))

)
