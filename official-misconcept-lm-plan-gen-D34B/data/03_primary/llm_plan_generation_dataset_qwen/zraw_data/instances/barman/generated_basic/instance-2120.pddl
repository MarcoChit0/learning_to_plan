(define (problem prob)
 (:domain barman)
 (:objects 
      shaker452 - shaker
      left right - hand
      shot80 shot235 shot118 - shot
      ingredient276 ingredient92 - ingredient
      cocktail422 - cocktail
      dispenser441 dispenser277 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker452)
  (ontable shot80)
  (ontable shot235)
  (ontable shot118)
  (dispenses dispenser441 ingredient276)
  (dispenses dispenser277 ingredient92)
  (clean shaker452)
  (clean shot80)
  (clean shot235)
  (clean shot118)
  (empty shaker452)
  (empty shot80)
  (empty shot235)
  (empty shot118)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker452 l0)
  (shaker-level shaker452 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient92)
  (cocktail-part2 cocktail422 ingredient276)
)
 (:goal
  (and
      (contains shot80 cocktail422)
      (contains shot235 ingredient276)
)))
