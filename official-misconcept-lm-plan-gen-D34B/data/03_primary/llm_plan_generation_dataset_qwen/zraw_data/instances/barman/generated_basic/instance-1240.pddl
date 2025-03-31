(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot72 shot492 shot464 - shot
      ingredient295 ingredient430 ingredient417 ingredient35 - ingredient
      cocktail108 - cocktail
      dispenser185 dispenser116 dispenser254 dispenser142 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot72)
  (ontable shot492)
  (ontable shot464)
  (dispenses dispenser185 ingredient295)
  (dispenses dispenser116 ingredient430)
  (dispenses dispenser254 ingredient417)
  (dispenses dispenser142 ingredient35)
  (clean shaker467)
  (clean shot72)
  (clean shot492)
  (clean shot464)
  (empty shaker467)
  (empty shot72)
  (empty shot492)
  (empty shot464)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail108 ingredient430)
  (cocktail-part2 cocktail108 ingredient417)
)
 (:goal
  (and
      (contains shot72 cocktail108)
      (contains shot492 cocktail108)
)))
