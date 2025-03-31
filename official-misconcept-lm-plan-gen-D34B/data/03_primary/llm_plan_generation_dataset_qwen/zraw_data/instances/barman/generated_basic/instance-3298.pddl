(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot157 - shot
      ingredient1 ingredient99 - ingredient
      cocktail258 - cocktail
      dispenser212 dispenser437 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot157)
  (dispenses dispenser212 ingredient1)
  (dispenses dispenser437 ingredient99)
  (clean shaker276)
  (clean shot157)
  (empty shaker276)
  (empty shot157)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail258 ingredient1)
  (cocktail-part2 cocktail258 ingredient99)
)
 (:goal
  (and
      (contains shot157 cocktail258)
)))
