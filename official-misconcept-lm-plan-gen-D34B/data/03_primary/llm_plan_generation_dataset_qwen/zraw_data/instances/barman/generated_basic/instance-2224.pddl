(define (problem prob)
 (:domain barman)
 (:objects 
      shaker60 - shaker
      left right - hand
      shot280 - shot
      ingredient296 ingredient479 - ingredient
      cocktail326 - cocktail
      dispenser242 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker60)
  (ontable shot280)
  (dispenses dispenser242 ingredient296)
  (dispenses dispenser458 ingredient479)
  (clean shaker60)
  (clean shot280)
  (empty shaker60)
  (empty shot280)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker60 l0)
  (shaker-level shaker60 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail326 ingredient296)
  (cocktail-part2 cocktail326 ingredient479)
)
 (:goal
  (and
      (contains shot280 cocktail326)
)))
