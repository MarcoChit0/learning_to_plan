(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot0 shot99 - shot
      ingredient360 ingredient315 - ingredient
      cocktail202 - cocktail
      dispenser298 dispenser4 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot0)
  (ontable shot99)
  (dispenses dispenser298 ingredient360)
  (dispenses dispenser4 ingredient315)
  (clean shaker398)
  (clean shot0)
  (clean shot99)
  (empty shaker398)
  (empty shot0)
  (empty shot99)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail202 ingredient360)
  (cocktail-part2 cocktail202 ingredient315)
)
 (:goal
  (and
      (contains shot0 cocktail202)
)))
