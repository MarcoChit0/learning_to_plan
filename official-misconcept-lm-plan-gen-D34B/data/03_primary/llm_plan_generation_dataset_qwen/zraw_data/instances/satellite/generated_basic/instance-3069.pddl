(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	infrared0 - mode
	star3 - direction
	groundstation2 - direction
	star0 - direction
	star1 - direction
	star4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite1 star4)
	(have_image star4 infrared0)
	(have_image phenomenon5 infrared0)
	(have_image star6 infrared0)
))

)
