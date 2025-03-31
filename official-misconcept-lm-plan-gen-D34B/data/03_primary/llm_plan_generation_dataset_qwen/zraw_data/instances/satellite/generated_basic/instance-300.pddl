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
	thermograph3 - mode
	infrared0 - mode
	thermograph1 - mode
	image2 - mode
	star1 - direction
	groundstation3 - direction
	groundstation2 - direction
	star5 - direction
	star4 - direction
	star0 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 thermograph3)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph3)
	(supports instrument4 thermograph1)
	(supports instrument4 image2)
	(calibration_target instrument4 star0)
	(calibration_target instrument4 star5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument5 thermograph3)
	(supports instrument5 image2)
	(calibration_target instrument5 star4)
	(supports instrument6 thermograph1)
	(supports instrument6 image2)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite1 phenomenon6)
	(have_image phenomenon6 infrared0)
))

)
