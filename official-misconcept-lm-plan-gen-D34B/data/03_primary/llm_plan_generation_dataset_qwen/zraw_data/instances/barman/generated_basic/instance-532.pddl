(define (problem prob)
 (:domain barman)
 (:objects 
      shaker95 - shaker
      left right - hand
      shot333 shot252 - shot
      ingredient286 ingredient89 - ingredient
      cocktail1 - cocktail
      dispenser171 dispenser195 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker95)
  (ontable shot333)
  (ontable shot252)
  (dispenses dispenser171 ingredient286)
  (dispenses dispenser195 ingredient89)
  (clean shaker95)
  (clean shot333)
  (clean shot252)
  (empty shaker95)
  (empty shot333)
  (empty shot252)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker95 l0)
  (shaker-level shaker95 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient286)
  (cocktail-part2 cocktail1 ingredient89)
)
 (:goal
  (and
      (contains shot333 cocktail1)
)))
