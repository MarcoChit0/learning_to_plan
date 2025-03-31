(define (problem prob)
 (:domain barman)
 (:objects 
      shaker395 - shaker
      left right - hand
      shot204 - shot
      ingredient451 ingredient207 ingredient157 - ingredient
      cocktail1 - cocktail
      dispenser110 dispenser71 dispenser81 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker395)
  (ontable shot204)
  (dispenses dispenser110 ingredient451)
  (dispenses dispenser71 ingredient207)
  (dispenses dispenser81 ingredient157)
  (clean shaker395)
  (clean shot204)
  (empty shaker395)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker395 l0)
  (shaker-level shaker395 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient157)
  (cocktail-part2 cocktail1 ingredient451)
)
 (:goal
  (and
      (contains shot204 cocktail1)
)))
