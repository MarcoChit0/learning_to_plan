(define (problem prob)
 (:domain barman)
 (:objects 
      shaker277 - shaker
      left right - hand
      shot425 shot3 shot244 - shot
      ingredient126 ingredient476 - ingredient
      cocktail466 - cocktail
      dispenser259 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker277)
  (ontable shot425)
  (ontable shot3)
  (ontable shot244)
  (dispenses dispenser259 ingredient126)
  (dispenses dispenser388 ingredient476)
  (clean shaker277)
  (clean shot425)
  (clean shot3)
  (clean shot244)
  (empty shaker277)
  (empty shot425)
  (empty shot3)
  (empty shot244)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker277 l0)
  (shaker-level shaker277 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail466 ingredient126)
  (cocktail-part2 cocktail466 ingredient476)
)
 (:goal
  (and
      (contains shot425 cocktail466)
      (contains shot3 cocktail466)
)))
