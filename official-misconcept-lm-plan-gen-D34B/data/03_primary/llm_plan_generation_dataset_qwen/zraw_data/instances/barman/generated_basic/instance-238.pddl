(define (problem prob)
 (:domain barman)
 (:objects 
      shaker142 - shaker
      left right - hand
      shot34 - shot
      ingredient177 ingredient77 - ingredient
      cocktail1 - cocktail
      dispenser28 dispenser477 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker142)
  (ontable shot34)
  (dispenses dispenser28 ingredient177)
  (dispenses dispenser477 ingredient77)
  (clean shaker142)
  (clean shot34)
  (empty shaker142)
  (empty shot34)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker142 l0)
  (shaker-level shaker142 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient177)
  (cocktail-part2 cocktail1 ingredient77)
)
 (:goal
  (and
      (contains shot34 cocktail1)
)))
