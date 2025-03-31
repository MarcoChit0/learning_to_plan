(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot448 shot340 - shot
      ingredient306 ingredient9 ingredient389 - ingredient
      cocktail1 - cocktail
      dispenser463 dispenser153 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot448)
  (ontable shot340)
  (dispenses dispenser463 ingredient306)
  (dispenses dispenser153 ingredient9)
  (dispenses dispenser274 ingredient389)
  (clean shaker360)
  (clean shot448)
  (clean shot340)
  (empty shaker360)
  (empty shot448)
  (empty shot340)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient306)
  (cocktail-part2 cocktail1 ingredient389)
)
 (:goal
  (and
      (contains shot448 cocktail1)
)))
