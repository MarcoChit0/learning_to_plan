(define (problem prob)
 (:domain barman)
 (:objects 
      shaker31 - shaker
      left right - hand
      shot134 shot280 shot361 - shot
      ingredient319 ingredient438 ingredient344 ingredient54 - ingredient
      cocktail216 - cocktail
      dispenser212 dispenser430 dispenser139 dispenser413 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker31)
  (ontable shot134)
  (ontable shot280)
  (ontable shot361)
  (dispenses dispenser212 ingredient319)
  (dispenses dispenser430 ingredient438)
  (dispenses dispenser139 ingredient344)
  (dispenses dispenser413 ingredient54)
  (clean shaker31)
  (clean shot134)
  (clean shot280)
  (clean shot361)
  (empty shaker31)
  (empty shot134)
  (empty shot280)
  (empty shot361)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker31 l0)
  (shaker-level shaker31 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail216 ingredient319)
  (cocktail-part2 cocktail216 ingredient344)
)
 (:goal
  (and
      (contains shot134 cocktail216)
      (contains shot280 cocktail216)
)))
