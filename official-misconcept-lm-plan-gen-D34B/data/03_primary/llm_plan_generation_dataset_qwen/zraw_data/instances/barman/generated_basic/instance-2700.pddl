(define (problem prob)
 (:domain barman)
 (:objects 
      shaker279 - shaker
      left right - hand
      shot419 shot205 shot34 - shot
      ingredient139 ingredient260 - ingredient
      cocktail216 - cocktail
      dispenser197 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker279)
  (ontable shot419)
  (ontable shot205)
  (ontable shot34)
  (dispenses dispenser197 ingredient139)
  (dispenses dispenser271 ingredient260)
  (clean shaker279)
  (clean shot419)
  (clean shot205)
  (clean shot34)
  (empty shaker279)
  (empty shot419)
  (empty shot205)
  (empty shot34)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker279 l0)
  (shaker-level shaker279 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail216 ingredient260)
  (cocktail-part2 cocktail216 ingredient139)
)
 (:goal
  (and
      (contains shot419 cocktail216)
      (contains shot205 cocktail216)
)))
