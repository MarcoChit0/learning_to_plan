(define (problem prob)
 (:domain barman)
 (:objects 
      shaker190 - shaker
      left right - hand
      shot216 - shot
      ingredient175 ingredient22 ingredient23 - ingredient
      cocktail198 - cocktail
      dispenser125 dispenser14 dispenser242 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker190)
  (ontable shot216)
  (dispenses dispenser125 ingredient175)
  (dispenses dispenser14 ingredient22)
  (dispenses dispenser242 ingredient23)
  (clean shaker190)
  (clean shot216)
  (empty shaker190)
  (empty shot216)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker190 l0)
  (shaker-level shaker190 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail198 ingredient23)
  (cocktail-part2 cocktail198 ingredient175)
)
 (:goal
  (and
      (contains shot216 cocktail198)
)))
