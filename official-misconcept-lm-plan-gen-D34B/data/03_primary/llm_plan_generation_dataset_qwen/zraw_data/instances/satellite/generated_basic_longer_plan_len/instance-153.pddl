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
	instrument5 - instrument
	instrument6 - instrument
	infrared2 - mode
	image3 - mode
	image1 - mode
	image0 - mode
	star1 - direction
	groundstation4 - direction
	star3 - direction
	star2 - direction
	groundstation0 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
	planet9 - direction
	planet10 - direction
	phenomenon11 - direction
	planet12 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 image1)
	(supports instrument1 infrared2)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon11)
	(supports instrument2 image0)
	(supports instrument2 image3)
	(calibration_target instrument2 star3)
	(supports instrument3 image0)
	(supports instrument3 image3)
	(calibration_target instrument3 star3)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared2)
	(supports instrument6 image0)
	(supports instrument6 image3)
	(calibration_target instrument6 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star8)
)
(:goal (and
	(pointing satellite1 star1)
	(have_image planet5 image0)
	(have_image star6 image3)
	(have_image phenomenon7 image0)
	(have_image star8 image1)
	(have_image planet9 image0)
	(have_image planet10 image3)
	(have_image phenomenon11 image1)
	(have_image planet12 image0)
))

)
