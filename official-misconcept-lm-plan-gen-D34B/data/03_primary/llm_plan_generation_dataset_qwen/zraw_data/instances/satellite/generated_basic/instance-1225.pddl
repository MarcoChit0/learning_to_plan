(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	spectrograph1 - mode
	infrared3 - mode
	spectrograph2 - mode
	infrared0 - mode
	groundstation1 - direction
	groundstation2 - direction
	groundstation0 - direction
	phenomenon3 - direction
	star4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph2)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument1 infrared0)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(have_image phenomenon3 infrared3)
	(have_image star4 infrared0)
	(have_image phenomenon5 spectrograph1)
))

)
