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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	infrared0 - mode
	groundstation2 - direction
	groundstation1 - direction
	star5 - direction
	star0 - direction
	star4 - direction
	star3 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star5)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star5)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star5)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star3)
	(calibration_target instrument7 star4)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 planet6)
	(pointing satellite3 planet6)
	(have_image planet6 infrared0)
))

)
