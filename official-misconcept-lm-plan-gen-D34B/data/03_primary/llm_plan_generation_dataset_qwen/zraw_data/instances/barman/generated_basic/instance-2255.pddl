(define (problem prob)
 (:domain barman)
 (:objects 
      shaker252 - shaker
      left right - hand
      shot60 shot486 shot414 - shot
      ingredient101 ingredient322 ingredient46 ingredient84 - ingredient
      cocktail458 - cocktail
      dispenser235 dispenser402 dispenser256 dispenser394 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker252)
  (ontable shot60)
  (ontable shot486)
  (ontable shot414)
  (dispenses dispenser235 ingredient101)
  (dispenses dispenser402 ingredient322)
  (dispenses dispenser256 ingredient46)
  (dispenses dispenser394 ingredient84)
  (clean shaker252)
  (clean shot60)
  (clean shot486)
  (clean shot414)
  (empty shaker252)
  (empty shot60)
  (empty shot486)
  (empty shot414)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker252 l0)
  (shaker-level shaker252 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail458 ingredient322)
  (cocktail-part2 cocktail458 ingredient46)
)
 (:goal
  (and
      (contains shot60 cocktail458)
      (contains shot486 ingredient84)
)))
