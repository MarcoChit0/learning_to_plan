(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	satellite4 - satellite
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	satellite5 - satellite
	instrument22 - instrument
	instrument23 - instrument
	infrared1 - mode
	image0 - mode
	groundstation2 - direction
	star1 - direction
	groundstation0 - direction
	star3 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(calibration_target instrument1 star3)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star3)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star1)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 infrared1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation2)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument11 infrared1)
	(calibration_target instrument11 star1)
	(supports instrument12 image0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 infrared1)
	(supports instrument13 image0)
	(calibration_target instrument13 star3)
	(supports instrument14 image0)
	(supports instrument14 infrared1)
	(calibration_target instrument14 star3)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
	(supports instrument15 image0)
	(supports instrument15 infrared1)
	(calibration_target instrument15 star1)
	(supports instrument16 image0)
	(calibration_target instrument16 star1)
	(supports instrument17 image0)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star3)
	(supports instrument18 infrared1)
	(calibration_target instrument18 groundstation0)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
	(supports instrument19 image0)
	(supports instrument19 infrared1)
	(calibration_target instrument19 groundstation0)
	(supports instrument20 image0)
	(supports instrument20 infrared1)
	(calibration_target instrument20 star3)
	(supports instrument21 infrared1)
	(supports instrument21 image0)
	(calibration_target instrument21 star1)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon5)
	(supports instrument22 infrared1)
	(supports instrument22 image0)
	(calibration_target instrument22 groundstation0)
	(supports instrument23 infrared1)
	(calibration_target instrument23 star3)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon5)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(pointing satellite1 planet4)
	(pointing satellite2 planet4)
	(pointing satellite3 groundstation2)
	(have_image planet4 infrared1)
	(have_image phenomenon5 image0)
))

)
