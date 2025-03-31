(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	satellite5 - satellite
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	image2 - mode
	infrared0 - mode
	image1 - mode
	groundstation2 - direction
	groundstation0 - direction
	star3 - direction
	star1 - direction
	planet4 - direction
	star5 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(supports instrument1 image2)
	(calibration_target instrument1 star1)
	(supports instrument2 image2)
	(calibration_target instrument2 star3)
	(supports instrument3 image2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument4 image2)
	(supports instrument4 infrared0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(supports instrument5 image2)
	(calibration_target instrument5 star1)
	(supports instrument6 image1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star3)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(supports instrument7 image2)
	(calibration_target instrument7 star3)
	(supports instrument8 image2)
	(calibration_target instrument8 star3)
	(supports instrument9 image1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star3)
	(supports instrument10 image2)
	(supports instrument10 infrared0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image2)
	(supports instrument12 infrared0)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument13 image2)
	(supports instrument13 image1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star1)
	(supports instrument14 image1)
	(calibration_target instrument14 star1)
	(supports instrument15 image2)
	(supports instrument15 image1)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 image1)
	(calibration_target instrument16 star3)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star3)
	(supports instrument18 infrared0)
	(supports instrument18 image1)
	(supports instrument18 image2)
	(calibration_target instrument18 star3)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet4)
	(supports instrument19 infrared0)
	(supports instrument19 image2)
	(calibration_target instrument19 star1)
	(supports instrument20 image2)
	(supports instrument20 image1)
	(supports instrument20 infrared0)
	(calibration_target instrument20 star1)
	(supports instrument21 image2)
	(calibration_target instrument21 star1)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star3)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 planet4)
	(have_image planet4 image2)
	(have_image star5 image2)
))

)
