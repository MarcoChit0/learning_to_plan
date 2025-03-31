(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	infrared0 - mode
	star4 - direction
	star0 - direction
	groundstation2 - direction
	star3 - direction
	star1 - direction
	groundstation5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 star1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(calibration_target instrument4 star4)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 star1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 star1)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite1 star4)
	(pointing satellite2 star0)
	(have_image phenomenon6 infrared0)
))

)
