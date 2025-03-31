(define (problem prob)
 (:domain barman)
 (:objects 
      shaker479 - shaker
      left right - hand
      shot45 shot438 shot253 - shot
      ingredient480 ingredient47 ingredient408 ingredient456 - ingredient
      cocktail1 - cocktail
      dispenser190 dispenser218 dispenser489 dispenser227 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker479)
  (ontable shot45)
  (ontable shot438)
  (ontable shot253)
  (dispenses dispenser190 ingredient480)
  (dispenses dispenser218 ingredient47)
  (dispenses dispenser489 ingredient408)
  (dispenses dispenser227 ingredient456)
  (clean shaker479)
  (clean shot45)
  (clean shot438)
  (clean shot253)
  (empty shaker479)
  (empty shot45)
  (empty shot438)
  (empty shot253)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker479 l0)
  (shaker-level shaker479 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient456)
  (cocktail-part2 cocktail1 ingredient47)
)
 (:goal
  (and
      (contains shot45 cocktail1)
      (contains shot438 ingredient47)
)))
