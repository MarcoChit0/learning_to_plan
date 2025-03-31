(define (problem prob)
 (:domain barman)
 (:objects 
      shaker335 - shaker
      left right - hand
      shot59 shot372 shot218 - shot
      ingredient247 ingredient308 ingredient114 ingredient331 - ingredient
      cocktail66 - cocktail
      dispenser410 dispenser465 dispenser263 dispenser209 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker335)
  (ontable shot59)
  (ontable shot372)
  (ontable shot218)
  (dispenses dispenser410 ingredient247)
  (dispenses dispenser465 ingredient308)
  (dispenses dispenser263 ingredient114)
  (dispenses dispenser209 ingredient331)
  (clean shaker335)
  (clean shot59)
  (clean shot372)
  (clean shot218)
  (empty shaker335)
  (empty shot59)
  (empty shot372)
  (empty shot218)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker335 l0)
  (shaker-level shaker335 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail66 ingredient114)
  (cocktail-part2 cocktail66 ingredient247)
)
 (:goal
  (and
      (contains shot59 cocktail66)
      (contains shot372 ingredient331)
)))
