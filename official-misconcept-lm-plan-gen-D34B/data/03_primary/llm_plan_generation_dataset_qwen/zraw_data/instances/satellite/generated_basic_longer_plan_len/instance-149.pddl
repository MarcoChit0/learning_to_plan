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
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image3 - mode
	thermograph0 - mode
	infrared1 - mode
	infrared2 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation2 - direction
	star3 - direction
	groundstation4 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
	star8 - direction
	phenomenon9 - direction
	phenomenon10 - direction
	planet11 - direction
	phenomenon12 - direction
)
(:init
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared2)
	(supports instrument1 image3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image3)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image3)
	(supports instrument4 infrared1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star8)
	(supports instrument6 infrared1)
	(supports instrument6 image3)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 image3)
	(supports instrument8 thermograph0)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation4)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star8)
)
(:goal (and
	(pointing satellite1 star8)
	(have_image planet5 infrared1)
	(have_image star6 infrared1)
	(have_image star7 thermograph0)
	(have_image star8 image3)
	(have_image phenomenon9 image3)
	(have_image phenomenon10 image3)
	(have_image planet11 infrared2)
	(have_image phenomenon12 infrared2)
))

)
