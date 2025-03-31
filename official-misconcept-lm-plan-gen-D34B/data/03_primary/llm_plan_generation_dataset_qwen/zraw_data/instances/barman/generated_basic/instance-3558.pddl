(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot142 shot0 - shot
      ingredient214 ingredient291 ingredient467 ingredient257 - ingredient
      cocktail173 - cocktail
      dispenser416 dispenser380 dispenser277 dispenser250 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot142)
  (ontable shot0)
  (dispenses dispenser416 ingredient214)
  (dispenses dispenser380 ingredient291)
  (dispenses dispenser277 ingredient467)
  (dispenses dispenser250 ingredient257)
  (clean shaker236)
  (clean shot142)
  (clean shot0)
  (empty shaker236)
  (empty shot142)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail173 ingredient214)
  (cocktail-part2 cocktail173 ingredient257)
)
 (:goal
  (and
      (contains shot142 cocktail173)
)))
