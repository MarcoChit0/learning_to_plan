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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	image1 - mode
	infrared2 - mode
	thermograph0 - mode
	star3 - direction
	star1 - direction
	groundstation4 - direction
	star5 - direction
	star0 - direction
	groundstation2 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(calibration_target instrument0 star5)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star5)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 star3)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star5)
	(calibration_target instrument6 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 infrared2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(calibration_target instrument8 star5)
	(supports instrument9 thermograph0)
	(supports instrument9 infrared2)
	(calibration_target instrument9 groundstation2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(have_image phenomenon6 image1)
	(have_image star7 thermograph0)
	(have_image planet8 infrared2)
	(have_image planet9 thermograph0)
))

)
