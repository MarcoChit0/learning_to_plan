(define (problem prob)
 (:domain barman)
 (:objects 
      shaker144 - shaker
      left right - hand
      shot104 shot302 - shot
      ingredient177 ingredient114 ingredient147 - ingredient
      cocktail273 - cocktail
      dispenser331 dispenser421 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker144)
  (ontable shot104)
  (ontable shot302)
  (dispenses dispenser331 ingredient177)
  (dispenses dispenser421 ingredient114)
  (dispenses dispenser100 ingredient147)
  (clean shaker144)
  (clean shot104)
  (clean shot302)
  (empty shaker144)
  (empty shot104)
  (empty shot302)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker144 l0)
  (shaker-level shaker144 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail273 ingredient177)
  (cocktail-part2 cocktail273 ingredient147)
)
 (:goal
  (and
      (contains shot104 cocktail273)
)))
