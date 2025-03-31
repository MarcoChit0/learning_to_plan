(define (problem prob)
 (:domain barman)
 (:objects 
      shaker18 - shaker
      left right - hand
      shot441 shot177 - shot
      ingredient366 ingredient165 - ingredient
      cocktail80 - cocktail
      dispenser388 dispenser98 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker18)
  (ontable shot441)
  (ontable shot177)
  (dispenses dispenser388 ingredient366)
  (dispenses dispenser98 ingredient165)
  (clean shaker18)
  (clean shot441)
  (clean shot177)
  (empty shaker18)
  (empty shot441)
  (empty shot177)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker18 l0)
  (shaker-level shaker18 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail80 ingredient165)
  (cocktail-part2 cocktail80 ingredient366)
)
 (:goal
  (and
      (contains shot441 cocktail80)
)))
