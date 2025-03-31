(define (problem prob)
 (:domain barman)
 (:objects 
      shaker433 - shaker
      left right - hand
      shot72 shot379 shot247 - shot
      ingredient52 ingredient33 ingredient465 ingredient417 - ingredient
      cocktail156 - cocktail
      dispenser70 dispenser417 dispenser287 dispenser473 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker433)
  (ontable shot72)
  (ontable shot379)
  (ontable shot247)
  (dispenses dispenser70 ingredient52)
  (dispenses dispenser417 ingredient33)
  (dispenses dispenser287 ingredient465)
  (dispenses dispenser473 ingredient417)
  (clean shaker433)
  (clean shot72)
  (clean shot379)
  (clean shot247)
  (empty shaker433)
  (empty shot72)
  (empty shot379)
  (empty shot247)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker433 l0)
  (shaker-level shaker433 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail156 ingredient417)
  (cocktail-part2 cocktail156 ingredient52)
)
 (:goal
  (and
      (contains shot72 cocktail156)
      (contains shot379 cocktail156)
)))
