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
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	image1 - mode
	infrared0 - mode
	star1 - direction
	groundstation4 - direction
	star3 - direction
	groundstation0 - direction
	groundstation2 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared0)
	(supports instrument3 image1)
	(calibration_target instrument3 star1)
	(supports instrument4 image1)
	(calibration_target instrument4 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star3)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
	(supports instrument10 image1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation2)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
)
(:goal (and
	(have_image star5 infrared0)
	(have_image star6 image1)
	(have_image phenomenon7 image1)
	(have_image star8 infrared0)
))

)
