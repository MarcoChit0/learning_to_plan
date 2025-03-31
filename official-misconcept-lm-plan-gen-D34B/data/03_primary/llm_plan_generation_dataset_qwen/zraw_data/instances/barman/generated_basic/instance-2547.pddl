(define (problem prob)
 (:domain barman)
 (:objects 
      shaker225 - shaker
      left right - hand
      shot203 shot41 - shot
      ingredient32 ingredient442 - ingredient
      cocktail236 - cocktail
      dispenser87 dispenser432 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker225)
  (ontable shot203)
  (ontable shot41)
  (dispenses dispenser87 ingredient32)
  (dispenses dispenser432 ingredient442)
  (clean shaker225)
  (clean shot203)
  (clean shot41)
  (empty shaker225)
  (empty shot203)
  (empty shot41)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker225 l0)
  (shaker-level shaker225 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail236 ingredient442)
  (cocktail-part2 cocktail236 ingredient32)
)
 (:goal
  (and
      (contains shot203 cocktail236)
)))
