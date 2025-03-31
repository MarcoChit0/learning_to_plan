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
	instrument5 - instrument
	instrument6 - instrument
	infrared0 - mode
	thermograph1 - mode
	groundstation0 - direction
	groundstation3 - direction
	groundstation2 - direction
	star1 - direction
	star4 - direction
	groundstation5 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star1)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 thermograph1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star4)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
)
(:goal (and
	(pointing satellite0 phenomenon7)
	(pointing satellite1 groundstation3)
	(have_image planet6 infrared0)
	(have_image phenomenon7 infrared0)
	(have_image phenomenon8 infrared0)
))

)
