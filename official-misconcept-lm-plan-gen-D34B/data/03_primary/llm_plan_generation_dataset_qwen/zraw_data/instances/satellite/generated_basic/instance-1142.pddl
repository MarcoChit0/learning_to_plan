(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite4 - satellite
	instrument6 - instrument
	instrument7 - instrument
	infrared1 - mode
	infrared2 - mode
	infrared0 - mode
	groundstation3 - direction
	star1 - direction
	groundstation0 - direction
	groundstation2 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared0)
	(supports instrument3 infrared2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation2)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star8)
	(supports instrument4 infrared2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation2)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument6 infrared0)
	(supports instrument6 infrared2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared0)
	(supports instrument7 infrared1)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation2)
	(on_board instrument6 satellite4)
	(on_board instrument7 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star4)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(pointing satellite3 star4)
	(have_image star4 infrared0)
	(have_image planet5 infrared1)
	(have_image planet6 infrared2)
	(have_image planet7 infrared1)
	(have_image star8 infrared0)
	(have_image phenomenon9 infrared0)
))

)
