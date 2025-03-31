(define (problem prob)
 (:domain barman)
 (:objects 
      shaker124 - shaker
      left right - hand
      shot368 shot89 shot245 - shot
      ingredient351 ingredient31 ingredient165 ingredient215 - ingredient
      cocktail175 - cocktail
      dispenser270 dispenser204 dispenser22 dispenser179 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker124)
  (ontable shot368)
  (ontable shot89)
  (ontable shot245)
  (dispenses dispenser270 ingredient351)
  (dispenses dispenser204 ingredient31)
  (dispenses dispenser22 ingredient165)
  (dispenses dispenser179 ingredient215)
  (clean shaker124)
  (clean shot368)
  (clean shot89)
  (clean shot245)
  (empty shaker124)
  (empty shot368)
  (empty shot89)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker124 l0)
  (shaker-level shaker124 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail175 ingredient351)
  (cocktail-part2 cocktail175 ingredient215)
)
 (:goal
  (and
      (contains shot368 cocktail175)
      (contains shot89 ingredient165)
)))
