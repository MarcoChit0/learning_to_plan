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
	thermograph1 - mode
	thermograph2 - mode
	thermograph0 - mode
	star5 - direction
	groundstation0 - direction
	groundstation1 - direction
	groundstation2 - direction
	star3 - direction
	groundstation4 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star5)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 thermograph2)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star5)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 star3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon7)
)
(:goal (and
	(pointing satellite0 phenomenon8)
	(pointing satellite1 star3)
	(pointing satellite2 phenomenon8)
	(have_image phenomenon6 thermograph1)
	(have_image phenomenon7 thermograph0)
	(have_image phenomenon8 thermograph2)
))

)
