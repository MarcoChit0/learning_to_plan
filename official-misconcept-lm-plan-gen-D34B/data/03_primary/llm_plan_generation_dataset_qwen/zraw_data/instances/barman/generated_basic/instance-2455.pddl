(define (problem prob)
 (:domain barman)
 (:objects 
      shaker83 - shaker
      left right - hand
      shot394 shot139 shot14 - shot
      ingredient65 ingredient398 ingredient192 ingredient165 - ingredient
      cocktail488 - cocktail
      dispenser425 dispenser367 dispenser226 dispenser194 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker83)
  (ontable shot394)
  (ontable shot139)
  (ontable shot14)
  (dispenses dispenser425 ingredient65)
  (dispenses dispenser367 ingredient398)
  (dispenses dispenser226 ingredient192)
  (dispenses dispenser194 ingredient165)
  (clean shaker83)
  (clean shot394)
  (clean shot139)
  (clean shot14)
  (empty shaker83)
  (empty shot394)
  (empty shot139)
  (empty shot14)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker83 l0)
  (shaker-level shaker83 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail488 ingredient65)
  (cocktail-part2 cocktail488 ingredient165)
)
 (:goal
  (and
      (contains shot394 cocktail488)
      (contains shot139 cocktail488)
)))
