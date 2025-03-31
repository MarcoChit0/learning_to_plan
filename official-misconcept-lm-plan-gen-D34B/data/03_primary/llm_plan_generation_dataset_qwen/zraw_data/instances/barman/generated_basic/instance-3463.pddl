(define (problem prob)
 (:domain barman)
 (:objects 
      shaker187 - shaker
      left right - hand
      shot270 shot448 shot320 - shot
      ingredient414 ingredient292 ingredient422 ingredient473 - ingredient
      cocktail100 - cocktail
      dispenser118 dispenser270 dispenser383 dispenser353 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker187)
  (ontable shot270)
  (ontable shot448)
  (ontable shot320)
  (dispenses dispenser118 ingredient414)
  (dispenses dispenser270 ingredient292)
  (dispenses dispenser383 ingredient422)
  (dispenses dispenser353 ingredient473)
  (clean shaker187)
  (clean shot270)
  (clean shot448)
  (clean shot320)
  (empty shaker187)
  (empty shot270)
  (empty shot448)
  (empty shot320)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker187 l0)
  (shaker-level shaker187 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail100 ingredient422)
  (cocktail-part2 cocktail100 ingredient292)
)
 (:goal
  (and
      (contains shot270 cocktail100)
      (contains shot448 ingredient422)
)))
