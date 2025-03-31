(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot307 shot241 - shot
      ingredient79 ingredient70 ingredient485 ingredient350 - ingredient
      cocktail169 - cocktail
      dispenser278 dispenser260 dispenser187 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot307)
  (ontable shot241)
  (dispenses dispenser278 ingredient79)
  (dispenses dispenser260 ingredient70)
  (dispenses dispenser187 ingredient485)
  (dispenses dispenser262 ingredient350)
  (clean shaker42)
  (clean shot307)
  (clean shot241)
  (empty shaker42)
  (empty shot307)
  (empty shot241)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail169 ingredient350)
  (cocktail-part2 cocktail169 ingredient79)
)
 (:goal
  (and
      (contains shot307 cocktail169)
)))
