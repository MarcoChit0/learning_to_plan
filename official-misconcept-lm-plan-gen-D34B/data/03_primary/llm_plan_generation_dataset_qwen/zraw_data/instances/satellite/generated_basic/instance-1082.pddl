(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	infrared2 - mode
	image1 - mode
	spectrograph0 - mode
	groundstation1 - direction
	groundstation3 - direction
	groundstation0 - direction
	groundstation2 - direction
	star4 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared2)
	(supports instrument1 image1)
	(calibration_target instrument1 star5)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image1)
	(supports instrument3 spectrograph0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 star4)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite1 phenomenon8)
	(have_image star6 infrared2)
	(have_image phenomenon7 infrared2)
	(have_image phenomenon8 infrared2)
))

)
