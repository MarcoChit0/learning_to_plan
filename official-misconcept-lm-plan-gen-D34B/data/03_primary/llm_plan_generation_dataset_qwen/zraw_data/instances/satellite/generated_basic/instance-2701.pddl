(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	spectrograph0 - mode
	image3 - mode
	infrared2 - mode
	spectrograph1 - mode
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	groundstation2 - direction
	star4 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 infrared2)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph1)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 groundstation2)
	(have_image star4 infrared2)
))

)
