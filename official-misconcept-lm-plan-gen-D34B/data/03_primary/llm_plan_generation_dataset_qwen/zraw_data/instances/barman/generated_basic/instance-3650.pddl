(define (problem prob)
 (:domain barman)
 (:objects 
      shaker60 - shaker
      left right - hand
      shot191 shot166 - shot
      ingredient328 ingredient139 ingredient230 ingredient413 - ingredient
      cocktail367 - cocktail
      dispenser336 dispenser48 dispenser157 dispenser487 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker60)
  (ontable shot191)
  (ontable shot166)
  (dispenses dispenser336 ingredient328)
  (dispenses dispenser48 ingredient139)
  (dispenses dispenser157 ingredient230)
  (dispenses dispenser487 ingredient413)
  (clean shaker60)
  (clean shot191)
  (clean shot166)
  (empty shaker60)
  (empty shot191)
  (empty shot166)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker60 l0)
  (shaker-level shaker60 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail367 ingredient328)
  (cocktail-part2 cocktail367 ingredient139)
)
 (:goal
  (and
      (contains shot191 cocktail367)
)))
