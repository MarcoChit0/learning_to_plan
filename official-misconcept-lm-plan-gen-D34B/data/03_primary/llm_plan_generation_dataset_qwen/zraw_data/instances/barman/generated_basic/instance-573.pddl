(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot485 shot116 - shot
      ingredient422 ingredient124 ingredient28 - ingredient
      cocktail1 - cocktail
      dispenser379 dispenser211 dispenser225 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot485)
  (ontable shot116)
  (dispenses dispenser379 ingredient422)
  (dispenses dispenser211 ingredient124)
  (dispenses dispenser225 ingredient28)
  (clean shaker314)
  (clean shot485)
  (clean shot116)
  (empty shaker314)
  (empty shot485)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient124)
  (cocktail-part2 cocktail1 ingredient422)
)
 (:goal
  (and
      (contains shot485 cocktail1)
)))
