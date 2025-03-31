(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image1 - mode
	infrared2 - mode
	infrared0 - mode
	groundstation2 - direction
	groundstation4 - direction
	groundstation3 - direction
	star5 - direction
	groundstation1 - direction
	groundstation0 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
	planet9 - direction
	star10 - direction
	star11 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(supports instrument1 image1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation4)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared0)
	(supports instrument6 image1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet9)
	(supports instrument8 image1)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation0)
	(calibration_target instrument8 groundstation1)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star8)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation0)
	(calibration_target instrument10 star5)
	(supports instrument11 infrared0)
	(supports instrument11 infrared2)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image1)
	(supports instrument12 infrared2)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation0)
	(calibration_target instrument12 groundstation1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite1 star8)
	(have_image planet6 infrared2)
	(have_image phenomenon7 image1)
	(have_image star8 image1)
	(have_image planet9 image1)
	(have_image star10 infrared2)
	(have_image star11 infrared2)
))

)
