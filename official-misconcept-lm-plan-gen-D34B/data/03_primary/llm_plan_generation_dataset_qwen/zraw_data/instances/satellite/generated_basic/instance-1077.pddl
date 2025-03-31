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
	infrared2 - mode
	infrared0 - mode
	thermograph1 - mode
	star2 - direction
	star5 - direction
	groundstation4 - direction
	groundstation1 - direction
	groundstation3 - direction
	star0 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 infrared2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 infrared0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument3 infrared2)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 infrared2)
	(supports instrument4 thermograph1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument5 thermograph1)
	(supports instrument5 infrared2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(pointing satellite1 phenomenon7)
	(have_image star6 infrared2)
	(have_image phenomenon7 thermograph1)
	(have_image star8 infrared2)
))

)
