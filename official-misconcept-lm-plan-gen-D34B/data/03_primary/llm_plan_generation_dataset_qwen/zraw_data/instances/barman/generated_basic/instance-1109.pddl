(define (problem prob)
 (:domain barman)
 (:objects 
      shaker327 - shaker
      left right - hand
      shot122 shot276 - shot
      ingredient479 ingredient369 - ingredient
      cocktail1 - cocktail
      dispenser284 dispenser199 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker327)
  (ontable shot122)
  (ontable shot276)
  (dispenses dispenser284 ingredient479)
  (dispenses dispenser199 ingredient369)
  (clean shaker327)
  (clean shot122)
  (clean shot276)
  (empty shaker327)
  (empty shot122)
  (empty shot276)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker327 l0)
  (shaker-level shaker327 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient479)
  (cocktail-part2 cocktail1 ingredient369)
)
 (:goal
  (and
      (contains shot122 cocktail1)
)))
