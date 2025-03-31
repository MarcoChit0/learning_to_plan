(define (problem prob)
 (:domain barman)
 (:objects 
      shaker142 - shaker
      left right - hand
      shot452 shot484 - shot
      ingredient106 ingredient242 ingredient116 - ingredient
      cocktail422 - cocktail
      dispenser81 dispenser188 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker142)
  (ontable shot452)
  (ontable shot484)
  (dispenses dispenser81 ingredient106)
  (dispenses dispenser188 ingredient242)
  (dispenses dispenser390 ingredient116)
  (clean shaker142)
  (clean shot452)
  (clean shot484)
  (empty shaker142)
  (empty shot452)
  (empty shot484)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker142 l0)
  (shaker-level shaker142 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient106)
  (cocktail-part2 cocktail422 ingredient242)
)
 (:goal
  (and
      (contains shot452 cocktail422)
)))
