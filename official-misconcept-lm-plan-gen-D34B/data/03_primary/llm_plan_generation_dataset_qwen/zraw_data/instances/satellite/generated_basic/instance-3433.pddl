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
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	satellite5 - satellite
	instrument20 - instrument
	infrared0 - mode
	image2 - mode
	infrared1 - mode
	groundstation0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	planet3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared0)
	(supports instrument5 infrared1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument6 infrared1)
	(supports instrument6 image2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared1)
	(supports instrument7 image2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image2)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 infrared1)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument11 image2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image2)
	(supports instrument12 infrared0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation0)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument13 image2)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation0)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument15 image2)
	(supports instrument15 infrared1)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 infrared0)
	(supports instrument17 image2)
	(supports instrument17 infrared1)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 image2)
	(supports instrument18 infrared0)
	(supports instrument18 infrared1)
	(calibration_target instrument18 groundstation1)
	(supports instrument19 infrared1)
	(supports instrument19 image2)
	(supports instrument19 infrared0)
	(calibration_target instrument19 groundstation1)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
	(supports instrument20 infrared1)
	(supports instrument20 infrared0)
	(calibration_target instrument20 groundstation1)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation1)
)
(:goal (and
	(pointing satellite2 planet4)
	(pointing satellite4 planet4)
	(have_image phenomenon2 infrared0)
	(have_image planet3 infrared1)
	(have_image planet4 infrared0)
))

)
