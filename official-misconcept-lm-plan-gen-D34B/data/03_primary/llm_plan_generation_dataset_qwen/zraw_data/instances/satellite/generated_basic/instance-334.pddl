(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	thermograph0 - mode
	image3 - mode
	infrared2 - mode
	thermograph1 - mode
	star2 - direction
	star4 - direction
	star0 - direction
	star3 - direction
	star1 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 image3)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 star4)
	(supports instrument1 infrared2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument2 infrared2)
	(supports instrument2 image3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star4)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(calibration_target instrument3 star4)
	(supports instrument4 thermograph1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star1)
	(supports instrument7 image3)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 star1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
)
(:goal (and
	(have_image phenomenon6 thermograph0)
))

)
