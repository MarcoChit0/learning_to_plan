(define (problem prob)
 (:domain barman)
 (:objects 
      shaker75 - shaker
      left right - hand
      shot316 shot145 - shot
      ingredient114 ingredient358 ingredient281 - ingredient
      cocktail444 - cocktail
      dispenser460 dispenser443 dispenser480 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker75)
  (ontable shot316)
  (ontable shot145)
  (dispenses dispenser460 ingredient114)
  (dispenses dispenser443 ingredient358)
  (dispenses dispenser480 ingredient281)
  (clean shaker75)
  (clean shot316)
  (clean shot145)
  (empty shaker75)
  (empty shot316)
  (empty shot145)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker75 l0)
  (shaker-level shaker75 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail444 ingredient358)
  (cocktail-part2 cocktail444 ingredient114)
)
 (:goal
  (and
      (contains shot316 cocktail444)
)))
