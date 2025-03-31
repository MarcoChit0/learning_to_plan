(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot155 shot14 - shot
      ingredient402 ingredient451 ingredient376 - ingredient
      cocktail238 - cocktail
      dispenser496 dispenser78 dispenser60 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot155)
  (ontable shot14)
  (dispenses dispenser496 ingredient402)
  (dispenses dispenser78 ingredient451)
  (dispenses dispenser60 ingredient376)
  (clean shaker42)
  (clean shot155)
  (clean shot14)
  (empty shaker42)
  (empty shot155)
  (empty shot14)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail238 ingredient376)
  (cocktail-part2 cocktail238 ingredient402)
)
 (:goal
  (and
      (contains shot155 cocktail238)
)))
