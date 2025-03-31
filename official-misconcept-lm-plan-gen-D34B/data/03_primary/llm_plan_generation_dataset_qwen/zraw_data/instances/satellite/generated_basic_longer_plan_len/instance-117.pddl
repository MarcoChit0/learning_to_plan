(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	infrared1 - mode
	thermograph0 - mode
	star1 - direction
	groundstation4 - direction
	groundstation2 - direction
	groundstation5 - direction
	star0 - direction
	star6 - direction
	star3 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	star9 - direction
	phenomenon10 - direction
	star11 - direction
	star12 - direction
	planet13 - direction
	star14 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 star6)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star12)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation4)
	(calibration_target instrument8 groundstation2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation2)
	(calibration_target instrument9 star6)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star3)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star0)
	(calibration_target instrument11 groundstation5)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star6)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star3)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star6)
)
(:goal (and
	(pointing satellite0 phenomenon8)
	(pointing satellite1 phenomenon7)
	(pointing satellite2 phenomenon8)
	(have_image phenomenon7 infrared1)
	(have_image phenomenon8 thermograph0)
	(have_image star9 thermograph0)
	(have_image phenomenon10 infrared1)
	(have_image star11 infrared1)
	(have_image star12 thermograph0)
	(have_image planet13 thermograph0)
	(have_image star14 thermograph0)
))

)
