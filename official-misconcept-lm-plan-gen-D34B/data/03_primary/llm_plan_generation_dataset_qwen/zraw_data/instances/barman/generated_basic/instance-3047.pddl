(define (problem prob)
 (:domain barman)
 (:objects 
      shaker74 - shaker
      left right - hand
      shot110 shot128 - shot
      ingredient168 ingredient145 ingredient7 - ingredient
      cocktail145 - cocktail
      dispenser487 dispenser95 dispenser495 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker74)
  (ontable shot110)
  (ontable shot128)
  (dispenses dispenser487 ingredient168)
  (dispenses dispenser95 ingredient145)
  (dispenses dispenser495 ingredient7)
  (clean shaker74)
  (clean shot110)
  (clean shot128)
  (empty shaker74)
  (empty shot110)
  (empty shot128)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker74 l0)
  (shaker-level shaker74 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail145 ingredient145)
  (cocktail-part2 cocktail145 ingredient168)
)
 (:goal
  (and
      (contains shot110 cocktail145)
)))
