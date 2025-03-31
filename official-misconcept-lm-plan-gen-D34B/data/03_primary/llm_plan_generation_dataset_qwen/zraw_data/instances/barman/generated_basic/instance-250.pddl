(define (problem prob)
 (:domain barman)
 (:objects 
      shaker152 - shaker
      left right - hand
      shot118 shot32 shot80 - shot
      ingredient211 ingredient321 ingredient360 - ingredient
      cocktail1 - cocktail
      dispenser79 dispenser0 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker152)
  (ontable shot118)
  (ontable shot32)
  (ontable shot80)
  (dispenses dispenser79 ingredient211)
  (dispenses dispenser0 ingredient321)
  (dispenses dispenser240 ingredient360)
  (clean shaker152)
  (clean shot118)
  (clean shot32)
  (clean shot80)
  (empty shaker152)
  (empty shot118)
  (empty shot32)
  (empty shot80)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker152 l0)
  (shaker-level shaker152 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient211)
  (cocktail-part2 cocktail1 ingredient360)
)
 (:goal
  (and
      (contains shot118 cocktail1)
      (contains shot32 cocktail1)
)))
