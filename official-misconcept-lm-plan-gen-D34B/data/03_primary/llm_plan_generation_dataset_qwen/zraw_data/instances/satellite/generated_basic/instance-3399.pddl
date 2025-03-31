(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image0 - mode
	thermograph1 - mode
	thermograph2 - mode
	star1 - direction
	star2 - direction
	groundstation0 - direction
	star3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star2)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star1)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph2)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star3)
	(supports instrument8 thermograph1)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 phenomenon4)
	(pointing satellite3 groundstation0)
	(have_image phenomenon4 image0)
))

)
