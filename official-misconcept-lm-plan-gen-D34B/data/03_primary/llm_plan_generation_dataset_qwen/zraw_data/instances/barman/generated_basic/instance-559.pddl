(define (problem prob)
 (:domain barman)
 (:objects 
      shaker199 - shaker
      left right - hand
      shot44 shot130 - shot
      ingredient26 ingredient491 ingredient369 - ingredient
      cocktail1 - cocktail
      dispenser312 dispenser468 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker199)
  (ontable shot44)
  (ontable shot130)
  (dispenses dispenser312 ingredient26)
  (dispenses dispenser468 ingredient491)
  (dispenses dispenser196 ingredient369)
  (clean shaker199)
  (clean shot44)
  (clean shot130)
  (empty shaker199)
  (empty shot44)
  (empty shot130)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker199 l0)
  (shaker-level shaker199 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient26)
  (cocktail-part2 cocktail1 ingredient369)
)
 (:goal
  (and
      (contains shot44 cocktail1)
)))
