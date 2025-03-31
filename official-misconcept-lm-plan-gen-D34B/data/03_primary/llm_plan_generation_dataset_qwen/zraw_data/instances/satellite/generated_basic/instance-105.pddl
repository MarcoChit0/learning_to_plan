(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	thermograph1 - mode
	thermograph0 - mode
	star2 - direction
	groundstation1 - direction
	groundstation0 - direction
	star3 - direction
	star5 - direction
	groundstation4 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation4)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 thermograph1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star5)
	(supports instrument6 thermograph1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation0)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star3)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation0)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
	(supports instrument10 thermograph1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star5)
	(calibration_target instrument10 star3)
	(supports instrument11 thermograph1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet7)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite4 star5)
	(have_image planet6 thermograph0)
	(have_image planet7 thermograph0)
	(have_image planet8 thermograph0)
	(have_image planet9 thermograph1)
))

)
