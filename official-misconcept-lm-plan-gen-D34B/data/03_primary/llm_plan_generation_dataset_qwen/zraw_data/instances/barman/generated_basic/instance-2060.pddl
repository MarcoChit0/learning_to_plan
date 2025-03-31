(define (problem prob)
 (:domain barman)
 (:objects 
      shaker115 - shaker
      left right - hand
      shot288 shot25 - shot
      ingredient471 ingredient295 - ingredient
      cocktail238 - cocktail
      dispenser236 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker115)
  (ontable shot288)
  (ontable shot25)
  (dispenses dispenser236 ingredient471)
  (dispenses dispenser150 ingredient295)
  (clean shaker115)
  (clean shot288)
  (clean shot25)
  (empty shaker115)
  (empty shot288)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker115 l0)
  (shaker-level shaker115 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail238 ingredient471)
  (cocktail-part2 cocktail238 ingredient295)
)
 (:goal
  (and
      (contains shot288 cocktail238)
)))
