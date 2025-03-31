(define (problem prob)
 (:domain barman)
 (:objects 
      shaker295 - shaker
      left right - hand
      shot35 shot89 - shot
      ingredient15 ingredient236 - ingredient
      cocktail1 - cocktail
      dispenser455 dispenser284 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker295)
  (ontable shot35)
  (ontable shot89)
  (dispenses dispenser455 ingredient15)
  (dispenses dispenser284 ingredient236)
  (clean shaker295)
  (clean shot35)
  (clean shot89)
  (empty shaker295)
  (empty shot35)
  (empty shot89)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker295 l0)
  (shaker-level shaker295 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient15)
  (cocktail-part2 cocktail1 ingredient236)
)
 (:goal
  (and
      (contains shot35 cocktail1)
)))
