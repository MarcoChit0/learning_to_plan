(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot171 shot203 - shot
      ingredient158 ingredient102 ingredient344 ingredient361 - ingredient
      cocktail178 - cocktail
      dispenser229 dispenser196 dispenser493 dispenser482 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot171)
  (ontable shot203)
  (dispenses dispenser229 ingredient158)
  (dispenses dispenser196 ingredient102)
  (dispenses dispenser493 ingredient344)
  (dispenses dispenser482 ingredient361)
  (clean shaker179)
  (clean shot171)
  (clean shot203)
  (empty shaker179)
  (empty shot171)
  (empty shot203)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail178 ingredient344)
  (cocktail-part2 cocktail178 ingredient102)
)
 (:goal
  (and
      (contains shot171 cocktail178)
)))
