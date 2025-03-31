(define (problem prob)
 (:domain barman)
 (:objects 
      shaker216 - shaker
      left right - hand
      shot473 shot145 shot183 - shot
      ingredient109 ingredient285 ingredient96 - ingredient
      cocktail386 - cocktail
      dispenser192 dispenser419 dispenser432 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker216)
  (ontable shot473)
  (ontable shot145)
  (ontable shot183)
  (dispenses dispenser192 ingredient109)
  (dispenses dispenser419 ingredient285)
  (dispenses dispenser432 ingredient96)
  (clean shaker216)
  (clean shot473)
  (clean shot145)
  (clean shot183)
  (empty shaker216)
  (empty shot473)
  (empty shot145)
  (empty shot183)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker216 l0)
  (shaker-level shaker216 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail386 ingredient109)
  (cocktail-part2 cocktail386 ingredient96)
)
 (:goal
  (and
      (contains shot473 cocktail386)
      (contains shot145 ingredient109)
)))
