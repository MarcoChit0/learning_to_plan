(define (problem prob)
 (:domain barman)
 (:objects 
      shaker404 - shaker
      left right - hand
      shot105 - shot
      ingredient137 ingredient243 - ingredient
      cocktail1 - cocktail
      dispenser313 dispenser251 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker404)
  (ontable shot105)
  (dispenses dispenser313 ingredient137)
  (dispenses dispenser251 ingredient243)
  (clean shaker404)
  (clean shot105)
  (empty shaker404)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker404 l0)
  (shaker-level shaker404 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient243)
  (cocktail-part2 cocktail1 ingredient137)
)
 (:goal
  (and
      (contains shot105 cocktail1)
)))
