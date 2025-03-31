(define (problem prob)
 (:domain barman)
 (:objects 
      shaker197 - shaker
      left right - hand
      shot41 - shot
      ingredient444 ingredient384 ingredient61 ingredient498 - ingredient
      cocktail355 - cocktail
      dispenser43 dispenser147 dispenser204 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker197)
  (ontable shot41)
  (dispenses dispenser43 ingredient444)
  (dispenses dispenser147 ingredient384)
  (dispenses dispenser204 ingredient61)
  (dispenses dispenser73 ingredient498)
  (clean shaker197)
  (clean shot41)
  (empty shaker197)
  (empty shot41)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker197 l0)
  (shaker-level shaker197 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail355 ingredient498)
  (cocktail-part2 cocktail355 ingredient384)
)
 (:goal
  (and
      (contains shot41 cocktail355)
)))
