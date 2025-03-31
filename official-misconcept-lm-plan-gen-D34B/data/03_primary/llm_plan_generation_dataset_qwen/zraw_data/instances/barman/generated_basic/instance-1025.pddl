(define (problem prob)
 (:domain barman)
 (:objects 
      shaker462 - shaker
      left right - hand
      shot134 - shot
      ingredient312 ingredient145 ingredient326 - ingredient
      cocktail1 - cocktail
      dispenser349 dispenser196 dispenser371 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker462)
  (ontable shot134)
  (dispenses dispenser349 ingredient312)
  (dispenses dispenser196 ingredient145)
  (dispenses dispenser371 ingredient326)
  (clean shaker462)
  (clean shot134)
  (empty shaker462)
  (empty shot134)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker462 l0)
  (shaker-level shaker462 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient145)
  (cocktail-part2 cocktail1 ingredient326)
)
 (:goal
  (and
      (contains shot134 cocktail1)
)))
