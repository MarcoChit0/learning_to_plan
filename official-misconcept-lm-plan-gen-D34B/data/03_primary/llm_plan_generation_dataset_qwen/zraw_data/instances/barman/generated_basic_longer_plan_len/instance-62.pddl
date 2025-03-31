(define (problem prob)
 (:domain barman)
 (:objects 
      shaker345 - shaker
      left right - hand
      shot82 shot379 shot390 shot106 - shot
      ingredient149 ingredient341 ingredient371 ingredient491 - ingredient
      cocktail1 - cocktail
      dispenser70 dispenser124 dispenser433 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker345)
  (ontable shot82)
  (ontable shot379)
  (ontable shot390)
  (ontable shot106)
  (dispenses dispenser70 ingredient149)
  (dispenses dispenser124 ingredient341)
  (dispenses dispenser433 ingredient371)
  (dispenses dispenser474 ingredient491)
  (clean shaker345)
  (clean shot82)
  (clean shot379)
  (clean shot390)
  (clean shot106)
  (empty shaker345)
  (empty shot82)
  (empty shot379)
  (empty shot390)
  (empty shot106)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker345 l0)
  (shaker-level shaker345 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient149)
  (cocktail-part2 cocktail1 ingredient491)
)
 (:goal
  (and
      (contains shot82 cocktail1)
      (contains shot379 ingredient371)
      (contains shot390 cocktail1)
)))
