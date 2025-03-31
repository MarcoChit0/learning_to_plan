(define (problem prob)
 (:domain barman)
 (:objects 
      shaker176 - shaker
      left right - hand
      shot498 - shot
      ingredient262 ingredient495 - ingredient
      cocktail302 - cocktail
      dispenser176 dispenser455 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker176)
  (ontable shot498)
  (dispenses dispenser176 ingredient262)
  (dispenses dispenser455 ingredient495)
  (clean shaker176)
  (clean shot498)
  (empty shaker176)
  (empty shot498)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker176 l0)
  (shaker-level shaker176 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail302 ingredient262)
  (cocktail-part2 cocktail302 ingredient495)
)
 (:goal
  (and
      (contains shot498 cocktail302)
)))
