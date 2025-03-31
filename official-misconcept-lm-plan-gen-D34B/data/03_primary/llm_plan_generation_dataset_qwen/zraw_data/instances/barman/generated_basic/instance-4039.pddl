(define (problem prob)
 (:domain barman)
 (:objects 
      shaker430 - shaker
      left right - hand
      shot129 shot17 - shot
      ingredient333 ingredient484 ingredient231 ingredient319 - ingredient
      cocktail407 - cocktail
      dispenser254 dispenser113 dispenser448 dispenser249 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker430)
  (ontable shot129)
  (ontable shot17)
  (dispenses dispenser254 ingredient333)
  (dispenses dispenser113 ingredient484)
  (dispenses dispenser448 ingredient231)
  (dispenses dispenser249 ingredient319)
  (clean shaker430)
  (clean shot129)
  (clean shot17)
  (empty shaker430)
  (empty shot129)
  (empty shot17)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker430 l0)
  (shaker-level shaker430 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail407 ingredient319)
  (cocktail-part2 cocktail407 ingredient231)
)
 (:goal
  (and
      (contains shot129 cocktail407)
)))
