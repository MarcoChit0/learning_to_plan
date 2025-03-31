(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	thermograph1 - mode
	image2 - mode
	infrared3 - mode
	infrared0 - mode
	star0 - direction
	groundstation2 - direction
	star3 - direction
	groundstation4 - direction
	groundstation1 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 infrared3)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared0)
	(supports instrument3 image2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument4 image2)
	(supports instrument4 infrared3)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 infrared3)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star3)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 infrared0)
	(supports instrument7 image2)
	(calibration_target instrument7 groundstation1)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite2 star3)
	(have_image phenomenon5 infrared3)
	(have_image phenomenon6 image2)
))

)
