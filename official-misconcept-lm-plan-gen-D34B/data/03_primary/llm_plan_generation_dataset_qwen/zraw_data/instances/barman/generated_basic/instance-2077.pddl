(define (problem prob)
 (:domain barman)
 (:objects 
      shaker458 - shaker
      left right - hand
      shot75 shot161 - shot
      ingredient482 ingredient213 - ingredient
      cocktail386 - cocktail
      dispenser179 dispenser171 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker458)
  (ontable shot75)
  (ontable shot161)
  (dispenses dispenser179 ingredient482)
  (dispenses dispenser171 ingredient213)
  (clean shaker458)
  (clean shot75)
  (clean shot161)
  (empty shaker458)
  (empty shot75)
  (empty shot161)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker458 l0)
  (shaker-level shaker458 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail386 ingredient482)
  (cocktail-part2 cocktail386 ingredient213)
)
 (:goal
  (and
      (contains shot75 cocktail386)
)))
