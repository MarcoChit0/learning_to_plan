(define (problem prob)
 (:domain barman)
 (:objects 
      shaker293 - shaker
      left right - hand
      shot474 shot355 - shot
      ingredient170 ingredient115 ingredient348 - ingredient
      cocktail1 - cocktail
      dispenser60 dispenser358 dispenser112 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker293)
  (ontable shot474)
  (ontable shot355)
  (dispenses dispenser60 ingredient170)
  (dispenses dispenser358 ingredient115)
  (dispenses dispenser112 ingredient348)
  (clean shaker293)
  (clean shot474)
  (clean shot355)
  (empty shaker293)
  (empty shot474)
  (empty shot355)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker293 l0)
  (shaker-level shaker293 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient115)
  (cocktail-part2 cocktail1 ingredient348)
)
 (:goal
  (and
      (contains shot474 cocktail1)
)))
