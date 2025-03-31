(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite5 - satellite
	instrument12 - instrument
	image1 - mode
	infrared0 - mode
	groundstation2 - direction
	star1 - direction
	star0 - direction
	planet3 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 infrared0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument8 image1)
	(calibration_target instrument8 star1)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
	(supports instrument9 image1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star0)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument10 image1)
	(calibration_target instrument10 star0)
	(supports instrument11 image1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star1)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet3)
	(supports instrument12 infrared0)
	(supports instrument12 image1)
	(calibration_target instrument12 star0)
	(on_board instrument12 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite3 planet3)
	(have_image planet3 infrared0)
	(have_image star4 image1)
	(have_image planet5 image1)
))

)
