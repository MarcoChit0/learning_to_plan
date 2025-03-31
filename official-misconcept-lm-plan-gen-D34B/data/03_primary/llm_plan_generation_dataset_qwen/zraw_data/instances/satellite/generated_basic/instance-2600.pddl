(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	infrared3 - mode
	infrared0 - mode
	thermograph1 - mode
	image2 - mode
	groundstation0 - direction
	star1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image2)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 infrared3)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(have_image phenomenon2 image2)
))

)
