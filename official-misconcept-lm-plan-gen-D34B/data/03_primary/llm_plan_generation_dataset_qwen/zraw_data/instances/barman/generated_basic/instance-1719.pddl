(define (problem prob)
 (:domain barman)
 (:objects 
      shaker138 - shaker
      left right - hand
      shot456 shot204 - shot
      ingredient298 ingredient123 ingredient160 ingredient85 - ingredient
      cocktail313 - cocktail
      dispenser377 dispenser279 dispenser418 dispenser34 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker138)
  (ontable shot456)
  (ontable shot204)
  (dispenses dispenser377 ingredient298)
  (dispenses dispenser279 ingredient123)
  (dispenses dispenser418 ingredient160)
  (dispenses dispenser34 ingredient85)
  (clean shaker138)
  (clean shot456)
  (clean shot204)
  (empty shaker138)
  (empty shot456)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker138 l0)
  (shaker-level shaker138 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail313 ingredient123)
  (cocktail-part2 cocktail313 ingredient85)
)
 (:goal
  (and
      (contains shot456 cocktail313)
)))
