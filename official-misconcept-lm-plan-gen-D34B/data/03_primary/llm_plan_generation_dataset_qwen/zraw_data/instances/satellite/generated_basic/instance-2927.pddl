(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	image3 - mode
	spectrograph2 - mode
	spectrograph1 - mode
	spectrograph0 - mode
	groundstation1 - direction
	groundstation0 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 spectrograph2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument3 image3)
	(supports instrument3 spectrograph2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
	(supports instrument5 spectrograph2)
	(supports instrument5 image3)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image3)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation0)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image3)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 spectrograph2)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 spectrograph2)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 spectrograph1)
	(calibration_target instrument16 groundstation0)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet3)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite2 phenomenon2)
	(have_image phenomenon2 spectrograph1)
	(have_image planet3 spectrograph1)
))

)
