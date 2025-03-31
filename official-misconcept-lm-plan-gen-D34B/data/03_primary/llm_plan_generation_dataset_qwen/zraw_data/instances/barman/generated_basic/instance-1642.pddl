(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot133 shot173 shot241 - shot
      ingredient437 ingredient339 ingredient323 ingredient190 - ingredient
      cocktail114 - cocktail
      dispenser322 dispenser454 dispenser39 dispenser259 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot133)
  (ontable shot173)
  (ontable shot241)
  (dispenses dispenser322 ingredient437)
  (dispenses dispenser454 ingredient339)
  (dispenses dispenser39 ingredient323)
  (dispenses dispenser259 ingredient190)
  (clean shaker266)
  (clean shot133)
  (clean shot173)
  (clean shot241)
  (empty shaker266)
  (empty shot133)
  (empty shot173)
  (empty shot241)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail114 ingredient190)
  (cocktail-part2 cocktail114 ingredient437)
)
 (:goal
  (and
      (contains shot133 cocktail114)
      (contains shot173 cocktail114)
)))
