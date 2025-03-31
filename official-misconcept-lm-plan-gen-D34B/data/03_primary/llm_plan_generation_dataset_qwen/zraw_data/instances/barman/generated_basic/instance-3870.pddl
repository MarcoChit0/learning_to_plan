(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot256 shot25 - shot
      ingredient122 ingredient237 ingredient79 - ingredient
      cocktail41 - cocktail
      dispenser99 dispenser306 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot256)
  (ontable shot25)
  (dispenses dispenser99 ingredient122)
  (dispenses dispenser306 ingredient237)
  (dispenses dispenser101 ingredient79)
  (clean shaker167)
  (clean shot256)
  (clean shot25)
  (empty shaker167)
  (empty shot256)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail41 ingredient237)
  (cocktail-part2 cocktail41 ingredient122)
)
 (:goal
  (and
      (contains shot256 cocktail41)
)))
