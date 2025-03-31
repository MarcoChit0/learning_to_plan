(define (problem prob)
 (:domain barman)
 (:objects 
      shaker91 - shaker
      left right - hand
      shot463 shot203 shot390 - shot
      ingredient377 ingredient439 - ingredient
      cocktail424 - cocktail
      dispenser100 dispenser293 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker91)
  (ontable shot463)
  (ontable shot203)
  (ontable shot390)
  (dispenses dispenser100 ingredient377)
  (dispenses dispenser293 ingredient439)
  (clean shaker91)
  (clean shot463)
  (clean shot203)
  (clean shot390)
  (empty shaker91)
  (empty shot463)
  (empty shot203)
  (empty shot390)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker91 l0)
  (shaker-level shaker91 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail424 ingredient439)
  (cocktail-part2 cocktail424 ingredient377)
)
 (:goal
  (and
      (contains shot463 cocktail424)
      (contains shot203 cocktail424)
)))
