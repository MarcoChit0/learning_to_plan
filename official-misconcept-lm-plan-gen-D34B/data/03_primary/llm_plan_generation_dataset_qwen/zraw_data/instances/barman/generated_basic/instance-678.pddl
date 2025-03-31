(define (problem prob)
 (:domain barman)
 (:objects 
      shaker29 - shaker
      left right - hand
      shot213 shot20 - shot
      ingredient415 ingredient158 - ingredient
      cocktail1 - cocktail
      dispenser79 dispenser301 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker29)
  (ontable shot213)
  (ontable shot20)
  (dispenses dispenser79 ingredient415)
  (dispenses dispenser301 ingredient158)
  (clean shaker29)
  (clean shot213)
  (clean shot20)
  (empty shaker29)
  (empty shot213)
  (empty shot20)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker29 l0)
  (shaker-level shaker29 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient158)
  (cocktail-part2 cocktail1 ingredient415)
)
 (:goal
  (and
      (contains shot213 cocktail1)
)))
