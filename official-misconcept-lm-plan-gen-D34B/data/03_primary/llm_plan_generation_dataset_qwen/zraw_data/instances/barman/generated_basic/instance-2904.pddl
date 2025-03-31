(define (problem prob)
 (:domain barman)
 (:objects 
      shaker463 - shaker
      left right - hand
      shot116 - shot
      ingredient115 ingredient99 - ingredient
      cocktail473 - cocktail
      dispenser325 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker463)
  (ontable shot116)
  (dispenses dispenser325 ingredient115)
  (dispenses dispenser465 ingredient99)
  (clean shaker463)
  (clean shot116)
  (empty shaker463)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker463 l0)
  (shaker-level shaker463 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail473 ingredient115)
  (cocktail-part2 cocktail473 ingredient99)
)
 (:goal
  (and
      (contains shot116 cocktail473)
)))
