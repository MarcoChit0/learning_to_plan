(define (problem prob)
 (:domain barman)
 (:objects 
      shaker363 - shaker
      left right - hand
      shot298 shot52 - shot
      ingredient73 ingredient168 - ingredient
      cocktail1 - cocktail
      dispenser148 dispenser453 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker363)
  (ontable shot298)
  (ontable shot52)
  (dispenses dispenser148 ingredient73)
  (dispenses dispenser453 ingredient168)
  (clean shaker363)
  (clean shot298)
  (clean shot52)
  (empty shaker363)
  (empty shot298)
  (empty shot52)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker363 l0)
  (shaker-level shaker363 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient168)
  (cocktail-part2 cocktail1 ingredient73)
)
 (:goal
  (and
      (contains shot298 cocktail1)
)))
