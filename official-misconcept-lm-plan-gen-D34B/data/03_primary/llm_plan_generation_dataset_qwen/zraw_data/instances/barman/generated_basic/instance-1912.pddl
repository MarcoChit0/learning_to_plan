(define (problem prob)
 (:domain barman)
 (:objects 
      shaker410 - shaker
      left right - hand
      shot79 - shot
      ingredient295 ingredient460 ingredient411 ingredient381 - ingredient
      cocktail93 - cocktail
      dispenser95 dispenser156 dispenser76 dispenser102 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker410)
  (ontable shot79)
  (dispenses dispenser95 ingredient295)
  (dispenses dispenser156 ingredient460)
  (dispenses dispenser76 ingredient411)
  (dispenses dispenser102 ingredient381)
  (clean shaker410)
  (clean shot79)
  (empty shaker410)
  (empty shot79)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker410 l0)
  (shaker-level shaker410 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail93 ingredient295)
  (cocktail-part2 cocktail93 ingredient381)
)
 (:goal
  (and
      (contains shot79 cocktail93)
)))
