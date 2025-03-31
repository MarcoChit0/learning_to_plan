(define (problem prob)
 (:domain barman)
 (:objects 
      shaker165 - shaker
      left right - hand
      shot368 shot138 - shot
      ingredient494 ingredient179 ingredient220 ingredient181 - ingredient
      cocktail1 - cocktail
      dispenser143 dispenser204 dispenser429 dispenser66 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker165)
  (ontable shot368)
  (ontable shot138)
  (dispenses dispenser143 ingredient494)
  (dispenses dispenser204 ingredient179)
  (dispenses dispenser429 ingredient220)
  (dispenses dispenser66 ingredient181)
  (clean shaker165)
  (clean shot368)
  (clean shot138)
  (empty shaker165)
  (empty shot368)
  (empty shot138)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker165 l0)
  (shaker-level shaker165 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient220)
  (cocktail-part2 cocktail1 ingredient181)
)
 (:goal
  (and
      (contains shot368 cocktail1)
)))
