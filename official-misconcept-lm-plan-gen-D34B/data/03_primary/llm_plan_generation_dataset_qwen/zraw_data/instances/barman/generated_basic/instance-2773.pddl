(define (problem prob)
 (:domain barman)
 (:objects 
      shaker410 - shaker
      left right - hand
      shot290 shot310 - shot
      ingredient237 ingredient304 ingredient92 - ingredient
      cocktail123 - cocktail
      dispenser339 dispenser74 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker410)
  (ontable shot290)
  (ontable shot310)
  (dispenses dispenser339 ingredient237)
  (dispenses dispenser74 ingredient304)
  (dispenses dispenser92 ingredient92)
  (clean shaker410)
  (clean shot290)
  (clean shot310)
  (empty shaker410)
  (empty shot290)
  (empty shot310)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker410 l0)
  (shaker-level shaker410 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail123 ingredient304)
  (cocktail-part2 cocktail123 ingredient92)
)
 (:goal
  (and
      (contains shot290 cocktail123)
)))
