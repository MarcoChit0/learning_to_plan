(define (problem prob)
 (:domain barman)
 (:objects 
      shaker261 - shaker
      left right - hand
      shot368 shot409 shot440 - shot
      ingredient312 ingredient107 ingredient48 ingredient4 - ingredient
      cocktail436 - cocktail
      dispenser203 dispenser84 dispenser379 dispenser166 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker261)
  (ontable shot368)
  (ontable shot409)
  (ontable shot440)
  (dispenses dispenser203 ingredient312)
  (dispenses dispenser84 ingredient107)
  (dispenses dispenser379 ingredient48)
  (dispenses dispenser166 ingredient4)
  (clean shaker261)
  (clean shot368)
  (clean shot409)
  (clean shot440)
  (empty shaker261)
  (empty shot368)
  (empty shot409)
  (empty shot440)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker261 l0)
  (shaker-level shaker261 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail436 ingredient48)
  (cocktail-part2 cocktail436 ingredient107)
)
 (:goal
  (and
      (contains shot368 cocktail436)
      (contains shot409 ingredient107)
)))
