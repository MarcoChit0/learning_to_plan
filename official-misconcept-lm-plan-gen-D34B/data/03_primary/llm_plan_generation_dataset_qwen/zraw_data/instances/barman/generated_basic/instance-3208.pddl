(define (problem prob)
 (:domain barman)
 (:objects 
      shaker105 - shaker
      left right - hand
      shot83 shot221 shot328 shot427 - shot
      ingredient344 ingredient172 ingredient318 - ingredient
      cocktail89 - cocktail
      dispenser80 dispenser312 dispenser63 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker105)
  (ontable shot83)
  (ontable shot221)
  (ontable shot328)
  (ontable shot427)
  (dispenses dispenser80 ingredient344)
  (dispenses dispenser312 ingredient172)
  (dispenses dispenser63 ingredient318)
  (clean shaker105)
  (clean shot83)
  (clean shot221)
  (clean shot328)
  (clean shot427)
  (empty shaker105)
  (empty shot83)
  (empty shot221)
  (empty shot328)
  (empty shot427)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker105 l0)
  (shaker-level shaker105 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail89 ingredient172)
  (cocktail-part2 cocktail89 ingredient344)
)
 (:goal
  (and
      (contains shot83 cocktail89)
      (contains shot221 ingredient344)
      (contains shot328 cocktail89)
)))
