(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot259 - shot
      ingredient296 ingredient75 ingredient499 ingredient230 - ingredient
      cocktail239 - cocktail
      dispenser205 dispenser67 dispenser445 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot259)
  (dispenses dispenser205 ingredient296)
  (dispenses dispenser67 ingredient75)
  (dispenses dispenser445 ingredient499)
  (dispenses dispenser431 ingredient230)
  (clean shaker136)
  (clean shot259)
  (empty shaker136)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail239 ingredient75)
  (cocktail-part2 cocktail239 ingredient499)
)
 (:goal
  (and
      (contains shot259 cocktail239)
)))
