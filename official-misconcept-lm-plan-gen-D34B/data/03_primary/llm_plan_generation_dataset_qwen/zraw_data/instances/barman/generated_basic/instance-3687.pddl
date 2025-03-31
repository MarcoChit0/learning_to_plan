(define (problem prob)
 (:domain barman)
 (:objects 
      shaker280 - shaker
      left right - hand
      shot221 shot102 shot165 - shot
      ingredient48 ingredient99 ingredient255 ingredient206 - ingredient
      cocktail134 - cocktail
      dispenser126 dispenser168 dispenser10 dispenser285 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker280)
  (ontable shot221)
  (ontable shot102)
  (ontable shot165)
  (dispenses dispenser126 ingredient48)
  (dispenses dispenser168 ingredient99)
  (dispenses dispenser10 ingredient255)
  (dispenses dispenser285 ingredient206)
  (clean shaker280)
  (clean shot221)
  (clean shot102)
  (clean shot165)
  (empty shaker280)
  (empty shot221)
  (empty shot102)
  (empty shot165)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker280 l0)
  (shaker-level shaker280 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail134 ingredient255)
  (cocktail-part2 cocktail134 ingredient99)
)
 (:goal
  (and
      (contains shot221 cocktail134)
      (contains shot102 ingredient255)
)))
