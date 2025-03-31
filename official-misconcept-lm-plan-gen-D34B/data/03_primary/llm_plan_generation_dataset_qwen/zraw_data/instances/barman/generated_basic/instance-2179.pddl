(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot126 shot138 - shot
      ingredient295 ingredient148 - ingredient
      cocktail358 - cocktail
      dispenser289 dispenser365 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot126)
  (ontable shot138)
  (dispenses dispenser289 ingredient295)
  (dispenses dispenser365 ingredient148)
  (clean shaker474)
  (clean shot126)
  (clean shot138)
  (empty shaker474)
  (empty shot126)
  (empty shot138)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail358 ingredient295)
  (cocktail-part2 cocktail358 ingredient148)
)
 (:goal
  (and
      (contains shot126 cocktail358)
)))
