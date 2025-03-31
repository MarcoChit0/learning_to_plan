(define (problem prob)
 (:domain barman)
 (:objects 
      shaker220 - shaker
      left right - hand
      shot488 shot74 - shot
      ingredient212 ingredient203 ingredient437 - ingredient
      cocktail1 - cocktail
      dispenser470 dispenser216 dispenser47 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker220)
  (ontable shot488)
  (ontable shot74)
  (dispenses dispenser470 ingredient212)
  (dispenses dispenser216 ingredient203)
  (dispenses dispenser47 ingredient437)
  (clean shaker220)
  (clean shot488)
  (clean shot74)
  (empty shaker220)
  (empty shot488)
  (empty shot74)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker220 l0)
  (shaker-level shaker220 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient437)
  (cocktail-part2 cocktail1 ingredient212)
)
 (:goal
  (and
      (contains shot488 cocktail1)
)))
