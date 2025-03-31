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
	infrared1 - mode
	infrared2 - mode
	thermograph3 - mode
	star0 - direction
	groundstation1 - direction
	groundstation2 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared1)
	(supports instrument4 infrared2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 infrared0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared0)
	(supports instrument6 infrared1)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
)
(:goal (and
	(pointing satellite0 planet3)
	(pointing satellite1 groundstation1)
	(have_image planet3 infrared2)
	(have_image star4 infrared2)
))

)
