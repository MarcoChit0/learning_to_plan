(define (problem prob)
 (:domain barman)
 (:objects 
      shaker41 - shaker
      left right - hand
      shot149 shot186 - shot
      ingredient136 ingredient266 ingredient425 ingredient422 - ingredient
      cocktail306 - cocktail
      dispenser146 dispenser406 dispenser195 dispenser395 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker41)
  (ontable shot149)
  (ontable shot186)
  (dispenses dispenser146 ingredient136)
  (dispenses dispenser406 ingredient266)
  (dispenses dispenser195 ingredient425)
  (dispenses dispenser395 ingredient422)
  (clean shaker41)
  (clean shot149)
  (clean shot186)
  (empty shaker41)
  (empty shot149)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker41 l0)
  (shaker-level shaker41 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail306 ingredient266)
  (cocktail-part2 cocktail306 ingredient425)
)
 (:goal
  (and
      (contains shot149 cocktail306)
)))
