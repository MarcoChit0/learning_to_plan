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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	infrared0 - mode
	thermograph1 - mode
	infrared2 - mode
	star1 - direction
	star0 - direction
	planet2 - direction
	planet3 - direction
	planet4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument5 infrared2)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star1)
	(supports instrument6 infrared2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
)
(:goal (and
	(have_image planet2 infrared2)
	(have_image planet3 infrared0)
	(have_image planet4 infrared0)
	(have_image planet5 infrared0)
	(have_image phenomenon6 infrared0)
))

)
