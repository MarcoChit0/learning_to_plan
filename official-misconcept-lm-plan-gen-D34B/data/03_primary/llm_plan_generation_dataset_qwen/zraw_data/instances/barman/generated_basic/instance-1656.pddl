(define (problem prob)
 (:domain barman)
 (:objects 
      shaker194 - shaker
      left right - hand
      shot66 shot219 shot84 - shot
      ingredient358 ingredient230 ingredient345 ingredient98 - ingredient
      cocktail16 - cocktail
      dispenser62 dispenser98 dispenser392 dispenser85 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker194)
  (ontable shot66)
  (ontable shot219)
  (ontable shot84)
  (dispenses dispenser62 ingredient358)
  (dispenses dispenser98 ingredient230)
  (dispenses dispenser392 ingredient345)
  (dispenses dispenser85 ingredient98)
  (clean shaker194)
  (clean shot66)
  (clean shot219)
  (clean shot84)
  (empty shaker194)
  (empty shot66)
  (empty shot219)
  (empty shot84)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker194 l0)
  (shaker-level shaker194 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail16 ingredient98)
  (cocktail-part2 cocktail16 ingredient358)
)
 (:goal
  (and
      (contains shot66 cocktail16)
      (contains shot219 cocktail16)
)))
