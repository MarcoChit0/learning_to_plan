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
	satellite4 - satellite
	instrument7 - instrument
	infrared0 - mode
	thermograph1 - mode
	star1 - direction
	groundstation2 - direction
	star4 - direction
	groundstation0 - direction
	star3 - direction
	groundstation5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation5)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation5)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument4 thermograph1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 star3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation5)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star3)
	(calibration_target instrument6 groundstation5)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument7 thermograph1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 star3)
	(on_board instrument7 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon7)
)
(:goal (and
	(pointing satellite3 groundstation0)
	(pointing satellite4 star9)
	(have_image star6 infrared0)
	(have_image phenomenon7 thermograph1)
	(have_image phenomenon8 thermograph1)
	(have_image star9 infrared0)
))

)
