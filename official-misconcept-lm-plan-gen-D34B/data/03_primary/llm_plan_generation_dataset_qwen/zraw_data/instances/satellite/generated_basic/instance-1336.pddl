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
	infrared1 - mode
	image3 - mode
	thermograph2 - mode
	image0 - mode
	star0 - direction
	star1 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 image3)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(supports instrument1 image3)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph2)
	(supports instrument2 image3)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared1)
	(supports instrument4 image0)
	(supports instrument4 image3)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument5 image3)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star0)
	(supports instrument6 image3)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(have_image star1 image0)
))

)
