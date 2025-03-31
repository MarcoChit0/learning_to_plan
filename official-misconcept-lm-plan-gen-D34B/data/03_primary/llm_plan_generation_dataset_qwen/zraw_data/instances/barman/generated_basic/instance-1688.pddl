(define (problem prob)
 (:domain barman)
 (:objects 
      shaker224 - shaker
      left right - hand
      shot40 shot181 shot214 shot481 - shot
      ingredient147 ingredient323 ingredient336 - ingredient
      cocktail207 - cocktail
      dispenser198 dispenser179 dispenser202 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker224)
  (ontable shot40)
  (ontable shot181)
  (ontable shot214)
  (ontable shot481)
  (dispenses dispenser198 ingredient147)
  (dispenses dispenser179 ingredient323)
  (dispenses dispenser202 ingredient336)
  (clean shaker224)
  (clean shot40)
  (clean shot181)
  (clean shot214)
  (clean shot481)
  (empty shaker224)
  (empty shot40)
  (empty shot181)
  (empty shot214)
  (empty shot481)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker224 l0)
  (shaker-level shaker224 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail207 ingredient147)
  (cocktail-part2 cocktail207 ingredient323)
)
 (:goal
  (and
      (contains shot40 cocktail207)
      (contains shot181 ingredient323)
      (contains shot214 cocktail207)
)))
