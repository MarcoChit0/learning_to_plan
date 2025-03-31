(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot133 shot416 shot287 - shot
      ingredient334 ingredient307 ingredient362 ingredient490 - ingredient
      cocktail124 - cocktail
      dispenser312 dispenser436 dispenser180 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot133)
  (ontable shot416)
  (ontable shot287)
  (dispenses dispenser312 ingredient334)
  (dispenses dispenser436 ingredient307)
  (dispenses dispenser180 ingredient362)
  (dispenses dispenser52 ingredient490)
  (clean shaker19)
  (clean shot133)
  (clean shot416)
  (clean shot287)
  (empty shaker19)
  (empty shot133)
  (empty shot416)
  (empty shot287)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail124 ingredient490)
  (cocktail-part2 cocktail124 ingredient307)
)
 (:goal
  (and
      (contains shot133 cocktail124)
      (contains shot416 cocktail124)
)))
