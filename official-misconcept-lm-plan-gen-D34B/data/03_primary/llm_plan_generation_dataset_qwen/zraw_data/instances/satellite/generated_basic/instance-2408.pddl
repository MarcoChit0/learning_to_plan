(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	thermograph0 - mode
	image2 - mode
	thermograph1 - mode
	infrared3 - mode
	star1 - direction
	star3 - direction
	star0 - direction
	star2 - direction
	groundstation4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument1 infrared3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared3)
	(supports instrument3 thermograph1)
	(supports instrument3 image2)
	(calibration_target instrument3 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument4 thermograph0)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star3)
	(supports instrument6 thermograph1)
	(supports instrument6 image2)
	(calibration_target instrument6 star3)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph0)
	(supports instrument8 infrared3)
	(supports instrument8 image2)
	(calibration_target instrument8 star2)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 thermograph1)
	(supports instrument10 infrared3)
	(calibration_target instrument10 groundstation4)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 star3)
	(pointing satellite3 star2)
	(have_image phenomenon5 thermograph0)
	(have_image planet6 thermograph1)
))

)
