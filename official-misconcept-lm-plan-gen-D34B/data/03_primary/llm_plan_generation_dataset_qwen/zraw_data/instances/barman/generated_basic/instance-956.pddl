(define (problem prob)
 (:domain barman)
 (:objects 
      shaker206 - shaker
      left right - hand
      shot59 - shot
      ingredient443 ingredient313 ingredient62 ingredient272 - ingredient
      cocktail1 - cocktail
      dispenser380 dispenser377 dispenser364 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker206)
  (ontable shot59)
  (dispenses dispenser380 ingredient443)
  (dispenses dispenser377 ingredient313)
  (dispenses dispenser364 ingredient62)
  (dispenses dispenser385 ingredient272)
  (clean shaker206)
  (clean shot59)
  (empty shaker206)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker206 l0)
  (shaker-level shaker206 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient313)
  (cocktail-part2 cocktail1 ingredient272)
)
 (:goal
  (and
      (contains shot59 cocktail1)
)))
