(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot28 - shot
      ingredient119 ingredient1 ingredient58 - ingredient
      cocktail66 - cocktail
      dispenser116 dispenser12 dispenser335 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot28)
  (dispenses dispenser116 ingredient119)
  (dispenses dispenser12 ingredient1)
  (dispenses dispenser335 ingredient58)
  (clean shaker469)
  (clean shot28)
  (empty shaker469)
  (empty shot28)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail66 ingredient1)
  (cocktail-part2 cocktail66 ingredient119)
)
 (:goal
  (and
      (contains shot28 cocktail66)
)))
