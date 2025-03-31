(define (problem prob)
 (:domain barman)
 (:objects 
      shaker395 - shaker
      left right - hand
      shot88 - shot
      ingredient333 ingredient214 ingredient444 ingredient32 - ingredient
      cocktail1 - cocktail
      dispenser402 dispenser272 dispenser426 dispenser35 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker395)
  (ontable shot88)
  (dispenses dispenser402 ingredient333)
  (dispenses dispenser272 ingredient214)
  (dispenses dispenser426 ingredient444)
  (dispenses dispenser35 ingredient32)
  (clean shaker395)
  (clean shot88)
  (empty shaker395)
  (empty shot88)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker395 l0)
  (shaker-level shaker395 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient333)
  (cocktail-part2 cocktail1 ingredient32)
)
 (:goal
  (and
      (contains shot88 cocktail1)
)))
