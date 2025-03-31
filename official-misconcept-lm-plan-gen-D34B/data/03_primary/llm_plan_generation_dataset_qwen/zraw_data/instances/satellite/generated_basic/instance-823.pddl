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
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	infrared1 - mode
	image0 - mode
	groundstation0 - direction
	star1 - direction
	star2 - direction
	star3 - direction
	planet4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument11 infrared1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 infrared1)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation0)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet5)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 image0)
	(supports instrument17 infrared1)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 infrared1)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 infrared1)
	(supports instrument19 image0)
	(calibration_target instrument19 groundstation0)
	(supports instrument20 infrared1)
	(supports instrument20 image0)
	(calibration_target instrument20 groundstation0)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet4)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite2 planet5)
	(pointing satellite4 star3)
	(have_image star1 image0)
	(have_image star2 image0)
	(have_image star3 infrared1)
	(have_image planet4 infrared1)
	(have_image planet5 image0)
))

)
