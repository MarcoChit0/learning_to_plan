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
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	infrared1 - mode
	image0 - mode
	groundstation1 - direction
	star3 - direction
	groundstation2 - direction
	groundstation0 - direction
	planet4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared1)
	(supports instrument7 image0)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument10 infrared1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 infrared1)
	(supports instrument13 image0)
	(calibration_target instrument13 star3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 infrared1)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 infrared1)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 infrared1)
	(supports instrument17 image0)
	(calibration_target instrument17 groundstation0)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
)
(:goal (and
	(pointing satellite2 star3)
	(pointing satellite3 star3)
	(have_image planet4 infrared1)
	(have_image planet5 image0)
	(have_image phenomenon6 infrared1)
))

)
