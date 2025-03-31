(define (problem prob)
 (:domain barman)
 (:objects 
      shaker129 - shaker
      left right - hand
      shot167 shot28 - shot
      ingredient172 ingredient230 ingredient96 - ingredient
      cocktail336 - cocktail
      dispenser351 dispenser60 dispenser491 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker129)
  (ontable shot167)
  (ontable shot28)
  (dispenses dispenser351 ingredient172)
  (dispenses dispenser60 ingredient230)
  (dispenses dispenser491 ingredient96)
  (clean shaker129)
  (clean shot167)
  (clean shot28)
  (empty shaker129)
  (empty shot167)
  (empty shot28)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker129 l0)
  (shaker-level shaker129 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail336 ingredient230)
  (cocktail-part2 cocktail336 ingredient96)
)
 (:goal
  (and
      (contains shot167 cocktail336)
)))
