(define (problem prob)
 (:domain barman)
 (:objects 
      shaker319 - shaker
      left right - hand
      shot314 shot163 shot407 - shot
      ingredient468 ingredient495 - ingredient
      cocktail395 - cocktail
      dispenser361 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker319)
  (ontable shot314)
  (ontable shot163)
  (ontable shot407)
  (dispenses dispenser361 ingredient468)
  (dispenses dispenser205 ingredient495)
  (clean shaker319)
  (clean shot314)
  (clean shot163)
  (clean shot407)
  (empty shaker319)
  (empty shot314)
  (empty shot163)
  (empty shot407)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker319 l0)
  (shaker-level shaker319 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail395 ingredient468)
  (cocktail-part2 cocktail395 ingredient495)
)
 (:goal
  (and
      (contains shot314 cocktail395)
      (contains shot163 cocktail395)
)))
