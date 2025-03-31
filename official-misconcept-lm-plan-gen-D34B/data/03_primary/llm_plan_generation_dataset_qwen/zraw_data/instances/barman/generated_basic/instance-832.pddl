(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot7 shot42 shot64 - shot
      ingredient311 ingredient111 ingredient200 ingredient76 - ingredient
      cocktail1 - cocktail
      dispenser259 dispenser261 dispenser413 dispenser83 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot7)
  (ontable shot42)
  (ontable shot64)
  (dispenses dispenser259 ingredient311)
  (dispenses dispenser261 ingredient111)
  (dispenses dispenser413 ingredient200)
  (dispenses dispenser83 ingredient76)
  (clean shaker305)
  (clean shot7)
  (clean shot42)
  (clean shot64)
  (empty shaker305)
  (empty shot7)
  (empty shot42)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient76)
  (cocktail-part2 cocktail1 ingredient311)
)
 (:goal
  (and
      (contains shot7 cocktail1)
      (contains shot42 ingredient200)
)))
