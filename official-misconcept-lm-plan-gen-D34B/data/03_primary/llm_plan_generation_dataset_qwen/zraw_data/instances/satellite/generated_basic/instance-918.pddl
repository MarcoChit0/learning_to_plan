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
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite4 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite5 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	infrared0 - mode
	infrared1 - mode
	groundstation0 - direction
	star3 - direction
	star1 - direction
	star2 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument4 infrared0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star3)
	(supports instrument6 infrared0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star1)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
	(supports instrument7 infrared0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star1)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation0)
	(on_board instrument7 satellite4)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
	(supports instrument10 infrared0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star1)
	(supports instrument11 infrared1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 infrared1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star2)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star3)
	(supports instrument14 infrared1)
	(calibration_target instrument14 star1)
	(supports instrument15 infrared0)
	(supports instrument15 infrared1)
	(calibration_target instrument15 star2)
	(on_board instrument10 satellite5)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star1)
)
(:goal (and
	(pointing satellite0 planet4)
	(pointing satellite3 groundstation0)
	(have_image planet4 infrared0)
	(have_image phenomenon5 infrared0)
))

)
