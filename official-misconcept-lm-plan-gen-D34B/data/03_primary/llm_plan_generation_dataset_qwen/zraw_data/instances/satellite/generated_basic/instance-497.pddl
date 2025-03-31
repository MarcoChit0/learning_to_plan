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
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	infrared1 - mode
	thermograph0 - mode
	thermograph2 - mode
	image3 - mode
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	groundstation4 - direction
	groundstation0 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star2)
	(supports instrument4 image3)
	(calibration_target instrument4 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument5 thermograph2)
	(supports instrument5 image3)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 thermograph0)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 infrared1)
	(supports instrument8 thermograph0)
	(supports instrument8 image3)
	(calibration_target instrument8 groundstation4)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument9 thermograph2)
	(supports instrument9 infrared1)
	(supports instrument9 image3)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 thermograph2)
	(supports instrument10 infrared1)
	(supports instrument10 image3)
	(calibration_target instrument10 groundstation0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite2 star6)
	(have_image star5 thermograph0)
	(have_image star6 thermograph2)
	(have_image phenomenon7 thermograph2)
))

)
