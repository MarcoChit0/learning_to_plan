(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot440 shot196 - shot
      ingredient330 ingredient43 ingredient102 ingredient327 - ingredient
      cocktail143 - cocktail
      dispenser34 dispenser18 dispenser404 dispenser226 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot440)
  (ontable shot196)
  (dispenses dispenser34 ingredient330)
  (dispenses dispenser18 ingredient43)
  (dispenses dispenser404 ingredient102)
  (dispenses dispenser226 ingredient327)
  (clean shaker273)
  (clean shot440)
  (clean shot196)
  (empty shaker273)
  (empty shot440)
  (empty shot196)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail143 ingredient102)
  (cocktail-part2 cocktail143 ingredient43)
)
 (:goal
  (and
      (contains shot440 cocktail143)
)))
