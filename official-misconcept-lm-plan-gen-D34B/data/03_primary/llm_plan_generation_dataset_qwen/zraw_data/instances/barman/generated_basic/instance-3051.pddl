(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot178 shot221 - shot
      ingredient342 ingredient128 ingredient419 - ingredient
      cocktail216 - cocktail
      dispenser86 dispenser280 dispenser471 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot178)
  (ontable shot221)
  (dispenses dispenser86 ingredient342)
  (dispenses dispenser280 ingredient128)
  (dispenses dispenser471 ingredient419)
  (clean shaker439)
  (clean shot178)
  (clean shot221)
  (empty shaker439)
  (empty shot178)
  (empty shot221)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail216 ingredient342)
  (cocktail-part2 cocktail216 ingredient128)
)
 (:goal
  (and
      (contains shot178 cocktail216)
)))
