(define (problem prob)
 (:domain barman)
 (:objects 
      shaker89 - shaker
      left right - hand
      shot206 shot435 - shot
      ingredient358 ingredient42 ingredient171 - ingredient
      cocktail112 - cocktail
      dispenser314 dispenser45 dispenser471 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker89)
  (ontable shot206)
  (ontable shot435)
  (dispenses dispenser314 ingredient358)
  (dispenses dispenser45 ingredient42)
  (dispenses dispenser471 ingredient171)
  (clean shaker89)
  (clean shot206)
  (clean shot435)
  (empty shaker89)
  (empty shot206)
  (empty shot435)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker89 l0)
  (shaker-level shaker89 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail112 ingredient171)
  (cocktail-part2 cocktail112 ingredient358)
)
 (:goal
  (and
      (contains shot206 cocktail112)
)))
