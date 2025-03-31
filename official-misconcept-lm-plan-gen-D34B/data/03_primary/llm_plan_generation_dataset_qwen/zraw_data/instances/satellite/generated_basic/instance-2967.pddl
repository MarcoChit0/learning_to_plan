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
	image1 - mode
	thermograph0 - mode
	thermograph2 - mode
	image3 - mode
	groundstation3 - direction
	star1 - direction
	star2 - direction
	star0 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 image3)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star2)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument4 thermograph2)
	(supports instrument4 image1)
	(supports instrument4 image3)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph2)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument6 thermograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star2)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(on_board instrument7 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation3)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 star1)
	(have_image planet4 thermograph2)
	(have_image phenomenon5 image3)
))

)
