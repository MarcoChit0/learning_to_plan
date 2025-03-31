(define (problem prob)
 (:domain barman)
 (:objects 
      shaker118 - shaker
      left right - hand
      shot280 - shot
      ingredient289 ingredient422 ingredient197 ingredient268 - ingredient
      cocktail1 - cocktail
      dispenser358 dispenser347 dispenser285 dispenser336 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker118)
  (ontable shot280)
  (dispenses dispenser358 ingredient289)
  (dispenses dispenser347 ingredient422)
  (dispenses dispenser285 ingredient197)
  (dispenses dispenser336 ingredient268)
  (clean shaker118)
  (clean shot280)
  (empty shaker118)
  (empty shot280)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker118 l0)
  (shaker-level shaker118 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient268)
  (cocktail-part2 cocktail1 ingredient422)
)
 (:goal
  (and
      (contains shot280 cocktail1)
)))
