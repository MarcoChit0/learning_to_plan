(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot336 shot228 - shot
      ingredient454 ingredient402 ingredient251 - ingredient
      cocktail138 - cocktail
      dispenser89 dispenser318 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot336)
  (ontable shot228)
  (dispenses dispenser89 ingredient454)
  (dispenses dispenser318 ingredient402)
  (dispenses dispenser130 ingredient251)
  (clean shaker472)
  (clean shot336)
  (clean shot228)
  (empty shaker472)
  (empty shot336)
  (empty shot228)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail138 ingredient251)
  (cocktail-part2 cocktail138 ingredient402)
)
 (:goal
  (and
      (contains shot336 cocktail138)
)))
