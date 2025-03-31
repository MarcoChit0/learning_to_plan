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
	instrument4 - instrument
	infrared1 - mode
	spectrograph3 - mode
	infrared2 - mode
	image0 - mode
	groundstation1 - direction
	star4 - direction
	star0 - direction
	groundstation2 - direction
	groundstation5 - direction
	groundstation3 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph3)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 star4)
	(supports instrument3 image0)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph3)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 groundstation5)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(pointing satellite1 groundstation5)
	(have_image phenomenon6 spectrograph3)
))

)
