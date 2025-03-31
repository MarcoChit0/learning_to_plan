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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	image0 - mode
	thermograph2 - mode
	infrared1 - mode
	groundstation3 - direction
	groundstation0 - direction
	groundstation2 - direction
	groundstation4 - direction
	star5 - direction
	groundstation1 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph2)
	(supports instrument0 image0)
	(calibration_target instrument0 star5)
	(supports instrument1 image0)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph2)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star5)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument8 infrared1)
	(supports instrument8 thermograph2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation4)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation1)
	(calibration_target instrument9 star5)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet8)
)
(:goal (and
	(pointing satellite0 phenomenon7)
	(pointing satellite1 groundstation1)
	(pointing satellite2 star5)
	(have_image planet6 thermograph2)
	(have_image phenomenon7 infrared1)
	(have_image planet8 infrared1)
	(have_image phenomenon9 image0)
))

)
