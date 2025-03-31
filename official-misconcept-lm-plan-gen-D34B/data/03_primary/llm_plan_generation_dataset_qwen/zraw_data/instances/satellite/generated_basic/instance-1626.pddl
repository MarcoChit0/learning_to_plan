(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	thermograph2 - mode
	infrared1 - mode
	infrared0 - mode
	star1 - direction
	star0 - direction
	phenomenon2 - direction
	phenomenon3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument5 infrared0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon3)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 star1)
	(have_image phenomenon2 thermograph2)
	(have_image phenomenon3 infrared0)
	(have_image planet4 infrared1)
))

)
