(define (problem prob)
 (:domain barman)
 (:objects 
      shaker49 - shaker
      left right - hand
      shot408 shot398 - shot
      ingredient231 ingredient288 ingredient76 - ingredient
      cocktail350 - cocktail
      dispenser121 dispenser462 dispenser499 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker49)
  (ontable shot408)
  (ontable shot398)
  (dispenses dispenser121 ingredient231)
  (dispenses dispenser462 ingredient288)
  (dispenses dispenser499 ingredient76)
  (clean shaker49)
  (clean shot408)
  (clean shot398)
  (empty shaker49)
  (empty shot408)
  (empty shot398)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker49 l0)
  (shaker-level shaker49 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail350 ingredient288)
  (cocktail-part2 cocktail350 ingredient231)
)
 (:goal
  (and
      (contains shot408 cocktail350)
)))
