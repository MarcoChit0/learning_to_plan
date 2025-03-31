(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	infrared0 - mode
	infrared1 - mode
	spectrograph2 - mode
	groundstation3 - direction
	star4 - direction
	groundstation2 - direction
	star1 - direction
	star0 - direction
	groundstation5 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation5)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation5)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star8)
)
(:goal (and
	(pointing satellite2 groundstation5)
	(have_image planet6 spectrograph2)
	(have_image planet7 spectrograph2)
	(have_image star8 infrared0)
))

)
