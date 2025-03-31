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
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	infrared0 - mode
	groundstation2 - direction
	star4 - direction
	star0 - direction
	star3 - direction
	star1 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star4)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star4)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star3)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(pointing satellite3 star4)
	(have_image planet5 infrared0)
	(have_image planet6 infrared0)
	(have_image phenomenon7 infrared0)
	(have_image phenomenon8 infrared0)
))

)
