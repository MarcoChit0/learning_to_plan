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
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	infrared0 - mode
	infrared2 - mode
	image1 - mode
	thermograph3 - mode
	groundstation1 - direction
	star2 - direction
	groundstation3 - direction
	star0 - direction
	groundstation4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument3 infrared2)
	(supports instrument3 thermograph3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation3)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument4 infrared2)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 thermograph3)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation3)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument6 thermograph3)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation4)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite3 phenomenon5)
	(have_image phenomenon5 thermograph3)
	(have_image planet6 thermograph3)
))

)
