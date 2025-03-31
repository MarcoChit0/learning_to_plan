(define (problem prob)
 (:domain barman)
 (:objects 
      shaker115 - shaker
      left right - hand
      shot144 shot148 - shot
      ingredient388 ingredient397 ingredient406 ingredient478 - ingredient
      cocktail375 - cocktail
      dispenser157 dispenser277 dispenser188 dispenser140 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker115)
  (ontable shot144)
  (ontable shot148)
  (dispenses dispenser157 ingredient388)
  (dispenses dispenser277 ingredient397)
  (dispenses dispenser188 ingredient406)
  (dispenses dispenser140 ingredient478)
  (clean shaker115)
  (clean shot144)
  (clean shot148)
  (empty shaker115)
  (empty shot144)
  (empty shot148)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker115 l0)
  (shaker-level shaker115 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail375 ingredient388)
  (cocktail-part2 cocktail375 ingredient406)
)
 (:goal
  (and
      (contains shot144 cocktail375)
)))
