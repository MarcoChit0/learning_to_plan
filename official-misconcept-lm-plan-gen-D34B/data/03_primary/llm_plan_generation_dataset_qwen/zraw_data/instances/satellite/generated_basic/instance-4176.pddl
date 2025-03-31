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
	spectrograph1 - mode
	spectrograph0 - mode
	star1 - direction
	star0 - direction
	groundstation2 - direction
	star3 - direction
	groundstation4 - direction
	groundstation5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 groundstation4)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(have_image planet6 spectrograph0)
))

)
