(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	infrared1 - mode
	image2 - mode
	infrared0 - mode
	star1 - direction
	groundstation5 - direction
	groundstation4 - direction
	groundstation2 - direction
	star0 - direction
	star3 - direction
	planet6 - direction
	star7 - direction
	star8 - direction
	star9 - direction
	planet10 - direction
	planet11 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(supports instrument1 infrared1)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation5)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star3)
	(supports instrument3 image2)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument4 infrared0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(calibration_target instrument4 star3)
	(supports instrument5 infrared1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star3)
	(supports instrument6 infrared0)
	(supports instrument6 infrared1)
	(supports instrument6 image2)
	(calibration_target instrument6 star0)
	(supports instrument7 image2)
	(calibration_target instrument7 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet11)
	(supports instrument8 infrared0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star3)
	(supports instrument9 image2)
	(supports instrument9 infrared1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation2)
	(calibration_target instrument9 star3)
	(supports instrument10 infrared1)
	(supports instrument10 image2)
	(calibration_target instrument10 star3)
	(supports instrument11 image2)
	(supports instrument11 infrared1)
	(calibration_target instrument11 star0)
	(supports instrument12 infrared0)
	(supports instrument12 image2)
	(calibration_target instrument12 star3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite2 star1)
	(have_image planet6 image2)
	(have_image star7 infrared0)
	(have_image star8 infrared0)
	(have_image star9 infrared1)
	(have_image planet10 infrared1)
	(have_image planet11 infrared1)
))

)
