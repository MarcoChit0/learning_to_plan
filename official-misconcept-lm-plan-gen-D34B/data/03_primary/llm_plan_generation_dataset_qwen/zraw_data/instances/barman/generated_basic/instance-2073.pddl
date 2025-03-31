(define (problem prob)
 (:domain barman)
 (:objects 
      shaker144 - shaker
      left right - hand
      shot284 shot49 - shot
      ingredient95 ingredient183 - ingredient
      cocktail84 - cocktail
      dispenser227 dispenser355 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker144)
  (ontable shot284)
  (ontable shot49)
  (dispenses dispenser227 ingredient95)
  (dispenses dispenser355 ingredient183)
  (clean shaker144)
  (clean shot284)
  (clean shot49)
  (empty shaker144)
  (empty shot284)
  (empty shot49)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker144 l0)
  (shaker-level shaker144 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail84 ingredient183)
  (cocktail-part2 cocktail84 ingredient95)
)
 (:goal
  (and
      (contains shot284 cocktail84)
)))
