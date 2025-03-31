(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	spectrograph0 - mode
	image3 - mode
	image2 - mode
	thermograph1 - mode
	groundstation2 - direction
	groundstation0 - direction
	star1 - direction
	planet3 - direction
	planet4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image3)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 image3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image3)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument5 spectrograph0)
	(supports instrument5 image2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph1)
	(supports instrument6 image3)
	(supports instrument6 image2)
	(calibration_target instrument6 star1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(have_image planet3 image2)
	(have_image planet4 spectrograph0)
	(have_image planet5 image3)
))

)
