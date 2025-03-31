(define (problem prob)
 (:domain barman)
 (:objects 
      shaker10 - shaker
      left right - hand
      shot141 - shot
      ingredient409 ingredient416 ingredient140 ingredient474 - ingredient
      cocktail148 - cocktail
      dispenser341 dispenser262 dispenser286 dispenser326 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker10)
  (ontable shot141)
  (dispenses dispenser341 ingredient409)
  (dispenses dispenser262 ingredient416)
  (dispenses dispenser286 ingredient140)
  (dispenses dispenser326 ingredient474)
  (clean shaker10)
  (clean shot141)
  (empty shaker10)
  (empty shot141)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker10 l0)
  (shaker-level shaker10 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail148 ingredient416)
  (cocktail-part2 cocktail148 ingredient140)
)
 (:goal
  (and
      (contains shot141 cocktail148)
)))
