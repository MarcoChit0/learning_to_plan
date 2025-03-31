(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot233 shot287 shot230 - shot
      ingredient382 ingredient168 ingredient441 - ingredient
      cocktail35 - cocktail
      dispenser47 dispenser106 dispenser135 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot233)
  (ontable shot287)
  (ontable shot230)
  (dispenses dispenser47 ingredient382)
  (dispenses dispenser106 ingredient168)
  (dispenses dispenser135 ingredient441)
  (clean shaker266)
  (clean shot233)
  (clean shot287)
  (clean shot230)
  (empty shaker266)
  (empty shot233)
  (empty shot287)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail35 ingredient382)
  (cocktail-part2 cocktail35 ingredient168)
)
 (:goal
  (and
      (contains shot233 cocktail35)
      (contains shot287 ingredient382)
)))
