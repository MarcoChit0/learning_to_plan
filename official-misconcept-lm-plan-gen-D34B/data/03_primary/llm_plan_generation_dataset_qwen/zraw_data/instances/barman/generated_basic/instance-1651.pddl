(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot319 shot482 shot345 - shot
      ingredient494 ingredient365 ingredient139 ingredient68 - ingredient
      cocktail136 - cocktail
      dispenser190 dispenser34 dispenser160 dispenser331 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot319)
  (ontable shot482)
  (ontable shot345)
  (dispenses dispenser190 ingredient494)
  (dispenses dispenser34 ingredient365)
  (dispenses dispenser160 ingredient139)
  (dispenses dispenser331 ingredient68)
  (clean shaker15)
  (clean shot319)
  (clean shot482)
  (clean shot345)
  (empty shaker15)
  (empty shot319)
  (empty shot482)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail136 ingredient365)
  (cocktail-part2 cocktail136 ingredient139)
)
 (:goal
  (and
      (contains shot319 cocktail136)
      (contains shot482 cocktail136)
)))
