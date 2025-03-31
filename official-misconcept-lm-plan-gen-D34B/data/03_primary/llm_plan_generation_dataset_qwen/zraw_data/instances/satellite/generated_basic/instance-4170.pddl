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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	infrared1 - mode
	image0 - mode
	groundstation4 - direction
	groundstation2 - direction
	star1 - direction
	groundstation5 - direction
	star0 - direction
	star3 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star1)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 star3)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star0)
	(calibration_target instrument7 groundstation2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument8 image0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation2)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 infrared1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation5)
	(calibration_target instrument9 star1)
	(supports instrument10 infrared1)
	(supports instrument10 image0)
	(calibration_target instrument10 star3)
	(calibration_target instrument10 star0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite0 planet6)
	(pointing satellite1 star0)
	(have_image planet6 infrared1)
))

)
