(define (problem prob)
 (:domain barman)
 (:objects 
      shaker268 - shaker
      left right - hand
      shot388 shot183 - shot
      ingredient350 ingredient145 ingredient371 - ingredient
      cocktail1 - cocktail
      dispenser247 dispenser24 dispenser169 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker268)
  (ontable shot388)
  (ontable shot183)
  (dispenses dispenser247 ingredient350)
  (dispenses dispenser24 ingredient145)
  (dispenses dispenser169 ingredient371)
  (clean shaker268)
  (clean shot388)
  (clean shot183)
  (empty shaker268)
  (empty shot388)
  (empty shot183)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker268 l0)
  (shaker-level shaker268 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient371)
  (cocktail-part2 cocktail1 ingredient350)
)
 (:goal
  (and
      (contains shot388 cocktail1)
)))
