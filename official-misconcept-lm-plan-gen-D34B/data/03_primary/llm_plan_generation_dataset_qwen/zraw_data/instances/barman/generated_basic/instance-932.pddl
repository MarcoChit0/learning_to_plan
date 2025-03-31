(define (problem prob)
 (:domain barman)
 (:objects 
      shaker181 - shaker
      left right - hand
      shot272 shot368 shot433 - shot
      ingredient247 ingredient289 ingredient158 ingredient323 - ingredient
      cocktail1 - cocktail
      dispenser334 dispenser458 dispenser105 dispenser333 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker181)
  (ontable shot272)
  (ontable shot368)
  (ontable shot433)
  (dispenses dispenser334 ingredient247)
  (dispenses dispenser458 ingredient289)
  (dispenses dispenser105 ingredient158)
  (dispenses dispenser333 ingredient323)
  (clean shaker181)
  (clean shot272)
  (clean shot368)
  (clean shot433)
  (empty shaker181)
  (empty shot272)
  (empty shot368)
  (empty shot433)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker181 l0)
  (shaker-level shaker181 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient289)
  (cocktail-part2 cocktail1 ingredient158)
)
 (:goal
  (and
      (contains shot272 cocktail1)
      (contains shot368 cocktail1)
)))
