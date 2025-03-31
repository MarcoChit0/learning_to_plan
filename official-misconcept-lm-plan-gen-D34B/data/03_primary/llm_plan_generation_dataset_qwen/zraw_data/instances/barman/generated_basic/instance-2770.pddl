(define (problem prob)
 (:domain barman)
 (:objects 
      shaker491 - shaker
      left right - hand
      shot185 shot319 - shot
      ingredient176 ingredient80 ingredient52 - ingredient
      cocktail365 - cocktail
      dispenser263 dispenser456 dispenser6 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker491)
  (ontable shot185)
  (ontable shot319)
  (dispenses dispenser263 ingredient176)
  (dispenses dispenser456 ingredient80)
  (dispenses dispenser6 ingredient52)
  (clean shaker491)
  (clean shot185)
  (clean shot319)
  (empty shaker491)
  (empty shot185)
  (empty shot319)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker491 l0)
  (shaker-level shaker491 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail365 ingredient176)
  (cocktail-part2 cocktail365 ingredient80)
)
 (:goal
  (and
      (contains shot185 cocktail365)
)))
