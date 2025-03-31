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
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	image0 - mode
	infrared1 - mode
	star3 - direction
	star1 - direction
	groundstation5 - direction
	groundstation0 - direction
	star2 - direction
	groundstation4 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation5)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star3)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star2)
	(calibration_target instrument7 groundstation0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument8 image0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation5)
	(supports instrument9 image0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation4)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star2)
	(calibration_target instrument10 groundstation4)
	(supports instrument11 infrared1)
	(supports instrument11 image0)
	(calibration_target instrument11 star2)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 infrared1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation0)
	(calibration_target instrument12 groundstation5)
	(supports instrument13 image0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 star2)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image0)
	(calibration_target instrument14 star2)
	(supports instrument15 image0)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
)
(:goal (and
	(have_image star6 image0)
	(have_image star7 infrared1)
	(have_image phenomenon8 image0)
	(have_image star9 infrared1)
))

)
