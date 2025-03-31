(define (problem prob)
 (:domain barman)
 (:objects 
      shaker152 - shaker
      left right - hand
      shot36 shot169 shot211 - shot
      ingredient287 ingredient35 ingredient406 - ingredient
      cocktail253 - cocktail
      dispenser425 dispenser223 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker152)
  (ontable shot36)
  (ontable shot169)
  (ontable shot211)
  (dispenses dispenser425 ingredient287)
  (dispenses dispenser223 ingredient35)
  (dispenses dispenser33 ingredient406)
  (clean shaker152)
  (clean shot36)
  (clean shot169)
  (clean shot211)
  (empty shaker152)
  (empty shot36)
  (empty shot169)
  (empty shot211)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker152 l0)
  (shaker-level shaker152 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail253 ingredient287)
  (cocktail-part2 cocktail253 ingredient35)
)
 (:goal
  (and
      (contains shot36 cocktail253)
      (contains shot169 cocktail253)
)))
