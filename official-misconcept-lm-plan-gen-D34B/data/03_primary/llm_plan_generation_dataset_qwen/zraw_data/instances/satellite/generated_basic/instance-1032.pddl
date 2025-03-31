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
	instrument5 - instrument
	instrument6 - instrument
	infrared0 - mode
	thermograph2 - mode
	thermograph1 - mode
	star3 - direction
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star3)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared0)
	(supports instrument6 thermograph1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
)
(:goal (and
	(have_image phenomenon4 thermograph1)
	(have_image phenomenon5 thermograph1)
	(have_image planet6 thermograph2)
	(have_image planet7 thermograph2)
	(have_image planet8 thermograph1)
))

)
