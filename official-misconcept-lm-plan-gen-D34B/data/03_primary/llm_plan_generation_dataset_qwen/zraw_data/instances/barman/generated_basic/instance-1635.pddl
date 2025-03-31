(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot253 shot151 shot420 - shot
      ingredient129 ingredient9 - ingredient
      cocktail227 - cocktail
      dispenser314 dispenser489 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot253)
  (ontable shot151)
  (ontable shot420)
  (dispenses dispenser314 ingredient129)
  (dispenses dispenser489 ingredient9)
  (clean shaker449)
  (clean shot253)
  (clean shot151)
  (clean shot420)
  (empty shaker449)
  (empty shot253)
  (empty shot151)
  (empty shot420)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail227 ingredient9)
  (cocktail-part2 cocktail227 ingredient129)
)
 (:goal
  (and
      (contains shot253 cocktail227)
      (contains shot151 cocktail227)
)))
