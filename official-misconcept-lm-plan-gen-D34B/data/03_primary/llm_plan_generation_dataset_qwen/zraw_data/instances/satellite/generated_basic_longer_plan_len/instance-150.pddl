(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	infrared2 - mode
	infrared0 - mode
	thermograph1 - mode
	thermograph3 - mode
	groundstation0 - direction
	groundstation4 - direction
	star3 - direction
	groundstation1 - direction
	groundstation2 - direction
	planet5 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
	star10 - direction
	phenomenon11 - direction
	star12 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph3)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 thermograph3)
	(supports instrument3 infrared2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph3)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared0)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 thermograph3)
	(supports instrument8 infrared2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star12)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite1 groundstation0)
	(have_image planet5 thermograph3)
	(have_image planet6 thermograph1)
	(have_image star7 thermograph1)
	(have_image planet8 thermograph3)
	(have_image planet9 thermograph1)
	(have_image star10 thermograph3)
	(have_image phenomenon11 thermograph1)
	(have_image star12 infrared0)
))

)
