(define (problem prob)
 (:domain barman)
 (:objects 
      shaker319 - shaker
      left right - hand
      shot66 shot434 - shot
      ingredient403 ingredient269 - ingredient
      cocktail179 - cocktail
      dispenser48 dispenser14 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker319)
  (ontable shot66)
  (ontable shot434)
  (dispenses dispenser48 ingredient403)
  (dispenses dispenser14 ingredient269)
  (clean shaker319)
  (clean shot66)
  (clean shot434)
  (empty shaker319)
  (empty shot66)
  (empty shot434)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker319 l0)
  (shaker-level shaker319 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail179 ingredient403)
  (cocktail-part2 cocktail179 ingredient269)
)
 (:goal
  (and
      (contains shot66 cocktail179)
)))
