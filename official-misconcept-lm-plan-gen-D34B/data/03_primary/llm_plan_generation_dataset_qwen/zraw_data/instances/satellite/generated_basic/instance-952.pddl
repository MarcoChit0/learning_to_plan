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
	instrument6 - instrument
	infrared2 - mode
	image3 - mode
	thermograph1 - mode
	thermograph0 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	star3 - direction
	phenomenon4 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star1)
	(supports instrument1 image3)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph1)
	(supports instrument2 infrared2)
	(supports instrument2 image3)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 image3)
	(supports instrument3 infrared2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph0)
	(supports instrument5 image3)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star3)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite2 phenomenon4)
	(have_image phenomenon4 thermograph0)
	(have_image star5 thermograph0)
	(have_image phenomenon6 infrared2)
))

)
