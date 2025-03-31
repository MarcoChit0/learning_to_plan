(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot382 shot139 - shot
      ingredient202 ingredient187 ingredient412 ingredient363 - ingredient
      cocktail72 - cocktail
      dispenser32 dispenser239 dispenser493 dispenser360 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot382)
  (ontable shot139)
  (dispenses dispenser32 ingredient202)
  (dispenses dispenser239 ingredient187)
  (dispenses dispenser493 ingredient412)
  (dispenses dispenser360 ingredient363)
  (clean shaker262)
  (clean shot382)
  (clean shot139)
  (empty shaker262)
  (empty shot382)
  (empty shot139)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail72 ingredient202)
  (cocktail-part2 cocktail72 ingredient187)
)
 (:goal
  (and
      (contains shot382 cocktail72)
)))
