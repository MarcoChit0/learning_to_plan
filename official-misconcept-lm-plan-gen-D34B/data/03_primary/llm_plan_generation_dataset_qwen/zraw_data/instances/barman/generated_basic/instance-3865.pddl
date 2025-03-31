(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot484 shot143 - shot
      ingredient454 ingredient113 ingredient394 - ingredient
      cocktail133 - cocktail
      dispenser73 dispenser402 dispenser135 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot484)
  (ontable shot143)
  (dispenses dispenser73 ingredient454)
  (dispenses dispenser402 ingredient113)
  (dispenses dispenser135 ingredient394)
  (clean shaker318)
  (clean shot484)
  (clean shot143)
  (empty shaker318)
  (empty shot484)
  (empty shot143)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail133 ingredient454)
  (cocktail-part2 cocktail133 ingredient113)
)
 (:goal
  (and
      (contains shot484 cocktail133)
)))
