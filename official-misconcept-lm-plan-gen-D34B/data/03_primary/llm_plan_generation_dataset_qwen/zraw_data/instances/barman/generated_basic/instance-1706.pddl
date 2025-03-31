(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot362 - shot
      ingredient177 ingredient169 ingredient259 ingredient371 - ingredient
      cocktail331 - cocktail
      dispenser178 dispenser8 dispenser223 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot362)
  (dispenses dispenser178 ingredient177)
  (dispenses dispenser8 ingredient169)
  (dispenses dispenser223 ingredient259)
  (dispenses dispenser101 ingredient371)
  (clean shaker236)
  (clean shot362)
  (empty shaker236)
  (empty shot362)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail331 ingredient169)
  (cocktail-part2 cocktail331 ingredient371)
)
 (:goal
  (and
      (contains shot362 cocktail331)
)))
