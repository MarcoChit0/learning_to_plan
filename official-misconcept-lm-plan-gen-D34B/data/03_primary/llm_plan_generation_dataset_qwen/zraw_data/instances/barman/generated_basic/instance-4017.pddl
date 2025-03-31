(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot12 shot189 - shot
      ingredient152 ingredient252 ingredient382 - ingredient
      cocktail205 - cocktail
      dispenser254 dispenser222 dispenser312 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot12)
  (ontable shot189)
  (dispenses dispenser254 ingredient152)
  (dispenses dispenser222 ingredient252)
  (dispenses dispenser312 ingredient382)
  (clean shaker126)
  (clean shot12)
  (clean shot189)
  (empty shaker126)
  (empty shot12)
  (empty shot189)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail205 ingredient152)
  (cocktail-part2 cocktail205 ingredient252)
)
 (:goal
  (and
      (contains shot12 cocktail205)
)))
