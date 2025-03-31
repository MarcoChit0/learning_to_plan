(define (problem prob)
 (:domain barman)
 (:objects 
      shaker413 - shaker
      left right - hand
      shot177 shot361 - shot
      ingredient257 ingredient196 ingredient48 ingredient217 - ingredient
      cocktail302 - cocktail
      dispenser358 dispenser318 dispenser378 dispenser380 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker413)
  (ontable shot177)
  (ontable shot361)
  (dispenses dispenser358 ingredient257)
  (dispenses dispenser318 ingredient196)
  (dispenses dispenser378 ingredient48)
  (dispenses dispenser380 ingredient217)
  (clean shaker413)
  (clean shot177)
  (clean shot361)
  (empty shaker413)
  (empty shot177)
  (empty shot361)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker413 l0)
  (shaker-level shaker413 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail302 ingredient217)
  (cocktail-part2 cocktail302 ingredient196)
)
 (:goal
  (and
      (contains shot177 cocktail302)
)))
