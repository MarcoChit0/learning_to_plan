(define (problem prob)
 (:domain barman)
 (:objects 
      shaker13 - shaker
      left right - hand
      shot323 shot50 - shot
      ingredient205 ingredient71 ingredient482 ingredient340 - ingredient
      cocktail274 - cocktail
      dispenser159 dispenser372 dispenser360 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker13)
  (ontable shot323)
  (ontable shot50)
  (dispenses dispenser159 ingredient205)
  (dispenses dispenser372 ingredient71)
  (dispenses dispenser360 ingredient482)
  (dispenses dispenser370 ingredient340)
  (clean shaker13)
  (clean shot323)
  (clean shot50)
  (empty shaker13)
  (empty shot323)
  (empty shot50)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker13 l0)
  (shaker-level shaker13 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail274 ingredient71)
  (cocktail-part2 cocktail274 ingredient482)
)
 (:goal
  (and
      (contains shot323 cocktail274)
)))
