(define (problem prob)
 (:domain barman)
 (:objects 
      shaker155 - shaker
      left right - hand
      shot349 shot110 shot436 - shot
      ingredient54 ingredient449 ingredient322 - ingredient
      cocktail411 - cocktail
      dispenser328 dispenser142 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker155)
  (ontable shot349)
  (ontable shot110)
  (ontable shot436)
  (dispenses dispenser328 ingredient54)
  (dispenses dispenser142 ingredient449)
  (dispenses dispenser214 ingredient322)
  (clean shaker155)
  (clean shot349)
  (clean shot110)
  (clean shot436)
  (empty shaker155)
  (empty shot349)
  (empty shot110)
  (empty shot436)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker155 l0)
  (shaker-level shaker155 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail411 ingredient322)
  (cocktail-part2 cocktail411 ingredient449)
)
 (:goal
  (and
      (contains shot349 cocktail411)
      (contains shot110 cocktail411)
)))
