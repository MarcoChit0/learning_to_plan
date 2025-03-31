(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite1 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	image0 - mode
	infrared1 - mode
	star4 - direction
	groundstation3 - direction
	star1 - direction
	groundstation2 - direction
	groundstation6 - direction
	groundstation0 - direction
	star5 - direction
	star7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	planet10 - direction
	star11 - direction
	star12 - direction
	phenomenon13 - direction
	star14 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star5)
	(supports instrument1 image0)
	(calibration_target instrument1 star4)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 star5)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 star1)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(on_board instrument6 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon13)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation6)
	(supports instrument8 image0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation0)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument9 image0)
	(calibration_target instrument9 star5)
	(calibration_target instrument9 groundstation0)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
)
(:goal (and
	(pointing satellite1 phenomenon8)
	(have_image star7 infrared1)
	(have_image phenomenon8 infrared1)
	(have_image phenomenon9 image0)
	(have_image planet10 infrared1)
	(have_image star11 infrared1)
	(have_image star12 infrared1)
	(have_image phenomenon13 infrared1)
	(have_image star14 image0)
))

)
