(define (problem prob)
 (:domain barman)
 (:objects 
      shaker143 - shaker
      left right - hand
      shot355 shot72 - shot
      ingredient53 ingredient407 ingredient3 - ingredient
      cocktail170 - cocktail
      dispenser77 dispenser403 dispenser383 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker143)
  (ontable shot355)
  (ontable shot72)
  (dispenses dispenser77 ingredient53)
  (dispenses dispenser403 ingredient407)
  (dispenses dispenser383 ingredient3)
  (clean shaker143)
  (clean shot355)
  (clean shot72)
  (empty shaker143)
  (empty shot355)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker143 l0)
  (shaker-level shaker143 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail170 ingredient407)
  (cocktail-part2 cocktail170 ingredient53)
)
 (:goal
  (and
      (contains shot355 cocktail170)
)))
