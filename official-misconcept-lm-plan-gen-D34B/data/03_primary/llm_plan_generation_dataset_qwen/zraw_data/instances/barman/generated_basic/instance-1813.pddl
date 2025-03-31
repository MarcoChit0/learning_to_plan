(define (problem prob)
 (:domain barman)
 (:objects 
      shaker328 - shaker
      left right - hand
      shot338 shot414 - shot
      ingredient129 ingredient150 ingredient482 ingredient352 - ingredient
      cocktail120 - cocktail
      dispenser195 dispenser85 dispenser261 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker328)
  (ontable shot338)
  (ontable shot414)
  (dispenses dispenser195 ingredient129)
  (dispenses dispenser85 ingredient150)
  (dispenses dispenser261 ingredient482)
  (dispenses dispenser474 ingredient352)
  (clean shaker328)
  (clean shot338)
  (clean shot414)
  (empty shaker328)
  (empty shot338)
  (empty shot414)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker328 l0)
  (shaker-level shaker328 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail120 ingredient129)
  (cocktail-part2 cocktail120 ingredient150)
)
 (:goal
  (and
      (contains shot338 cocktail120)
)))
