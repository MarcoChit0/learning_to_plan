(define (problem prob)
 (:domain barman)
 (:objects 
      shaker206 - shaker
      left right - hand
      shot280 shot438 - shot
      ingredient196 ingredient352 ingredient91 ingredient365 - ingredient
      cocktail1 - cocktail
      dispenser394 dispenser339 dispenser224 dispenser266 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker206)
  (ontable shot280)
  (ontable shot438)
  (dispenses dispenser394 ingredient196)
  (dispenses dispenser339 ingredient352)
  (dispenses dispenser224 ingredient91)
  (dispenses dispenser266 ingredient365)
  (clean shaker206)
  (clean shot280)
  (clean shot438)
  (empty shaker206)
  (empty shot280)
  (empty shot438)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker206 l0)
  (shaker-level shaker206 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient196)
  (cocktail-part2 cocktail1 ingredient352)
)
 (:goal
  (and
      (contains shot280 cocktail1)
)))
