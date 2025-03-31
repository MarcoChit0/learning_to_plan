(define (problem prob)
 (:domain barman)
 (:objects 
      shaker438 - shaker
      left right - hand
      shot319 shot228 - shot
      ingredient40 ingredient164 ingredient296 ingredient198 - ingredient
      cocktail147 - cocktail
      dispenser442 dispenser51 dispenser498 dispenser324 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker438)
  (ontable shot319)
  (ontable shot228)
  (dispenses dispenser442 ingredient40)
  (dispenses dispenser51 ingredient164)
  (dispenses dispenser498 ingredient296)
  (dispenses dispenser324 ingredient198)
  (clean shaker438)
  (clean shot319)
  (clean shot228)
  (empty shaker438)
  (empty shot319)
  (empty shot228)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker438 l0)
  (shaker-level shaker438 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail147 ingredient40)
  (cocktail-part2 cocktail147 ingredient164)
)
 (:goal
  (and
      (contains shot319 cocktail147)
)))
