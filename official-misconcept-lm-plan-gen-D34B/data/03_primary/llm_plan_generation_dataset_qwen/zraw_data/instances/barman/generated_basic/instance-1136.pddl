(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot232 shot228 - shot
      ingredient165 ingredient402 ingredient173 - ingredient
      cocktail1 - cocktail
      dispenser33 dispenser247 dispenser401 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot232)
  (ontable shot228)
  (dispenses dispenser33 ingredient165)
  (dispenses dispenser247 ingredient402)
  (dispenses dispenser401 ingredient173)
  (clean shaker488)
  (clean shot232)
  (clean shot228)
  (empty shaker488)
  (empty shot232)
  (empty shot228)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient165)
  (cocktail-part2 cocktail1 ingredient173)
)
 (:goal
  (and
      (contains shot232 cocktail1)
)))
