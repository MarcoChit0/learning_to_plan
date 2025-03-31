(define (problem prob)
 (:domain barman)
 (:objects 
      shaker278 - shaker
      left right - hand
      shot269 shot48 shot357 - shot
      ingredient455 ingredient451 - ingredient
      cocktail227 - cocktail
      dispenser106 dispenser366 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker278)
  (ontable shot269)
  (ontable shot48)
  (ontable shot357)
  (dispenses dispenser106 ingredient455)
  (dispenses dispenser366 ingredient451)
  (clean shaker278)
  (clean shot269)
  (clean shot48)
  (clean shot357)
  (empty shaker278)
  (empty shot269)
  (empty shot48)
  (empty shot357)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker278 l0)
  (shaker-level shaker278 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail227 ingredient455)
  (cocktail-part2 cocktail227 ingredient451)
)
 (:goal
  (and
      (contains shot269 cocktail227)
      (contains shot48 cocktail227)
)))
