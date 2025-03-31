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
	image0 - mode
	image2 - mode
	thermograph1 - mode
	groundstation5 - direction
	star2 - direction
	star0 - direction
	star1 - direction
	star3 - direction
	star4 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 image2)
	(calibration_target instrument3 star1)
	(supports instrument4 image2)
	(supports instrument4 image0)
	(calibration_target instrument4 star3)
	(calibration_target instrument4 star4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star3)
	(supports instrument6 thermograph1)
	(supports instrument6 image0)
	(supports instrument6 image2)
	(calibration_target instrument6 star4)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(pointing satellite1 star1)
	(pointing satellite2 star0)
	(have_image phenomenon6 thermograph1)
))

)
