(define (problem prob)
 (:domain barman)
 (:objects 
      shaker25 - shaker
      left right - hand
      shot16 shot409 - shot
      ingredient18 ingredient404 ingredient416 - ingredient
      cocktail337 - cocktail
      dispenser232 dispenser176 dispenser158 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker25)
  (ontable shot16)
  (ontable shot409)
  (dispenses dispenser232 ingredient18)
  (dispenses dispenser176 ingredient404)
  (dispenses dispenser158 ingredient416)
  (clean shaker25)
  (clean shot16)
  (clean shot409)
  (empty shaker25)
  (empty shot16)
  (empty shot409)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker25 l0)
  (shaker-level shaker25 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail337 ingredient18)
  (cocktail-part2 cocktail337 ingredient416)
)
 (:goal
  (and
      (contains shot16 cocktail337)
)))
