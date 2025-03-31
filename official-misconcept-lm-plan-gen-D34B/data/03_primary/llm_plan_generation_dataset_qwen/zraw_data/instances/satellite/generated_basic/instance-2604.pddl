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
	instrument7 - instrument
	thermograph1 - mode
	image3 - mode
	image2 - mode
	thermograph0 - mode
	groundstation1 - direction
	star0 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph1)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image3)
	(supports instrument2 image2)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 thermograph1)
	(supports instrument3 image3)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(supports instrument6 thermograph1)
	(supports instrument6 image2)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(have_image phenomenon2 image2)
))

)
