(define (problem prob)
 (:domain barman)
 (:objects 
      shaker290 - shaker
      left right - hand
      shot470 shot36 - shot
      ingredient43 ingredient300 ingredient202 - ingredient
      cocktail94 - cocktail
      dispenser44 dispenser466 dispenser358 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker290)
  (ontable shot470)
  (ontable shot36)
  (dispenses dispenser44 ingredient43)
  (dispenses dispenser466 ingredient300)
  (dispenses dispenser358 ingredient202)
  (clean shaker290)
  (clean shot470)
  (clean shot36)
  (empty shaker290)
  (empty shot470)
  (empty shot36)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker290 l0)
  (shaker-level shaker290 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail94 ingredient202)
  (cocktail-part2 cocktail94 ingredient43)
)
 (:goal
  (and
      (contains shot470 cocktail94)
)))
