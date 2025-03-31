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
	thermograph0 - mode
	groundstation0 - direction
	star2 - direction
	groundstation5 - direction
	star3 - direction
	groundstation4 - direction
	star1 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star1)
	(calibration_target instrument1 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation0)
	(calibration_target instrument3 star3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star1)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation5)
	(calibration_target instrument8 star1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star3)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star1)
	(calibration_target instrument10 groundstation4)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(pointing satellite2 star1)
	(pointing satellite3 groundstation4)
	(have_image planet6 thermograph0)
))

)
