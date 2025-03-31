(define (problem prob)
 (:domain barman)
 (:objects 
      shaker225 - shaker
      left right - hand
      shot146 shot33 - shot
      ingredient127 ingredient385 ingredient354 ingredient19 - ingredient
      cocktail216 - cocktail
      dispenser496 dispenser156 dispenser461 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker225)
  (ontable shot146)
  (ontable shot33)
  (dispenses dispenser496 ingredient127)
  (dispenses dispenser156 ingredient385)
  (dispenses dispenser461 ingredient354)
  (dispenses dispenser40 ingredient19)
  (clean shaker225)
  (clean shot146)
  (clean shot33)
  (empty shaker225)
  (empty shot146)
  (empty shot33)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker225 l0)
  (shaker-level shaker225 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail216 ingredient385)
  (cocktail-part2 cocktail216 ingredient127)
)
 (:goal
  (and
      (contains shot146 cocktail216)
)))
