(define (problem prob)
 (:domain barman)
 (:objects 
      shaker101 - shaker
      left right - hand
      shot462 shot105 - shot
      ingredient376 ingredient88 - ingredient
      cocktail31 - cocktail
      dispenser175 dispenser139 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker101)
  (ontable shot462)
  (ontable shot105)
  (dispenses dispenser175 ingredient376)
  (dispenses dispenser139 ingredient88)
  (clean shaker101)
  (clean shot462)
  (clean shot105)
  (empty shaker101)
  (empty shot462)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker101 l0)
  (shaker-level shaker101 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail31 ingredient376)
  (cocktail-part2 cocktail31 ingredient88)
)
 (:goal
  (and
      (contains shot462 cocktail31)
)))
