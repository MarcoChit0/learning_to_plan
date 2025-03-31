(define (problem prob)
 (:domain barman)
 (:objects 
      shaker376 - shaker
      left right - hand
      shot487 shot0 shot77 - shot
      ingredient242 ingredient63 - ingredient
      cocktail122 - cocktail
      dispenser43 dispenser302 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker376)
  (ontable shot487)
  (ontable shot0)
  (ontable shot77)
  (dispenses dispenser43 ingredient242)
  (dispenses dispenser302 ingredient63)
  (clean shaker376)
  (clean shot487)
  (clean shot0)
  (clean shot77)
  (empty shaker376)
  (empty shot487)
  (empty shot0)
  (empty shot77)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker376 l0)
  (shaker-level shaker376 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail122 ingredient63)
  (cocktail-part2 cocktail122 ingredient242)
)
 (:goal
  (and
      (contains shot487 cocktail122)
      (contains shot0 cocktail122)
)))
